import argparse
import pandas as pd
import itertools
import os

from ast import literal_eval

from helper_functions import *


# converts command-line flag to boolean
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "y")


# Computes basic stats for a data segment; takes in a Pandas Dataframe with
# appropriate fields (sum_costs and num_comorbidities).
def segment_stats(data):
  row_dict = {}
  row_dict['avg_cost'] = data.sum_costs.mean()
  row_dict['std_cost'] = data.sum_costs.std()
  row_dict['num_person_years'] = data.shape[0]
  row_dict['avg_num_com'] = data.no_comorbidities.mean()
  row_dict['std_num_com'] = data.no_comorbidities.std()
    
  return row_dict


def main(args):
  df_merged = pd.read_csv(args.input_file)
  df_merged.classes = df_merged.classes.apply(lambda x: literal_eval(x))
  df_merged.agg_indices = \
    df_merged.agg_indices.fillna('None').apply(lambda x: literal_eval(x))

  df_merged.sex.fillna('Not provided')
  pop_N = df_merged.shape[0] # grab total number of person-years

  # IMPORT DESCRIPTION INDEX
  desc_dict = pd.read_excel(args.disease_dict).long.to_dict()

  # We will be grouping by pairs of comorbid diseases.
  disease_dyads = [i for i in itertools.combinations(range(len(desc_dict)), 2)]
  # Ancillary function to convert pairs of disease indices to pairs of
  # disease descriptions.
  def map_dyad(d_dyad):
    return [desc_dict[d_dyad[0]], desc_dict[d_dyad[1]]]

  bins = [0,18,35,50,65,150]
  age_labels = ['0-18','18-35','35-50','50-65','65+']
  df_merged['age_bin'] = pd.cut(df_merged.age, bins=bins, labels=age_labels)

  sexes = ['F','M']

  # convert comorbidity list to boolean columns
  ## comorbidity_indicator()
  for i in range(len(desc_dict)):
    df_merged[desc_dict[i]] = comorbidity_indicator(df_merged,[str(i)])

  # disease stats
  diseases_df = pd.DataFrame()
  for i in range(len(desc_dict)):
    temp = df_merged[df_merged[desc_dict[i]]]
    diseases_df = diseases_df.append(pd.Series(segment_stats(temp), name=i))
  diseases_df['frequency'] = diseases_df.num_person_years / pop_N
#  diseases_df.to_csv(os.path.join(args.output_path, 'diseases_all_pop.csv'))
    
  # comorbid disease dyad stats
  d_dyad_df = pd.DataFrame()

  for i in disease_dyads:
    d_dyad = map_dyad(i)
    d_dyad_segment = df_merged[df_merged[d_dyad[0]] & df_merged[d_dyad[1]]]
    d_dyad_df = d_dyad_df.append(
      pd.Series(segment_stats(d_dyad_segment), name=i)
    )

  expected_df = pd.DataFrame()

  for i in d_dyad_df.index.tolist():
    expected_df = expected_df.append(pd.Series(
      {'expected': (diseases_df.iloc[i[0]].frequency *\
                    diseases_df.iloc[i[1]].frequency) },
      name = i
    ))
  
    
  d_dyad_df = pd.concat([d_dyad_df, expected_df], axis = 1)
  d_dyad_df['observed'] = d_dyad_df.num_person_years / pop_N
  d_dyad_df['O/E'] = d_dyad_df.observed / d_dyad_df.expected
  d_dyad_df['diseases'] = d_dyad_df.index.map(map_dyad)
  d_dyad_df['total_cost'] = d_dyad_df.avg_cost * d_dyad_df.num_person_years

  # Filter by minimum number of person-years
  d_dyad_df = d_dyad_df[d_dyad_df.num_person_years >= args.py_filter]
  
  d_dyad_df.sort_values(by=args.sort_type, ascending=False).to_csv(
    os.path.join(args.output_path, 'disease_dyads_all_pop.csv'),
    index=False, float_format='%.2f'
  )
  
  if not (args.age_bucket or args.gender_bucket): return
  # diseases + population segments
  buckets = [list(map(lambda x: map_dyad(x), disease_dyads))]
  if args.age_bucket:
    buckets.append(age_labels)
  if args.gender_bucket:
    buckets.append(sexes)
  segments = list(itertools.product(*buckets))

  segments_df = pd.DataFrame()

  for idx, segment in enumerate(segments):
    segment_df = df_merged[df_merged[segment[0][0]] & df_merged[segment[0][1]]]
    if args.age_bucket:
      segment_df = segment_df[segment_df.age_bin == segment[1]]
    if args.gender_bucket:
      segment_df = segment_df[segment_df.sex == segment[-1]]
    stats = segment_stats(segment_df)
    stats['segment'] = segment
    segments_df = segments_df.append(pd.Series(stats, name=idx))

  # Filter by minimum number of person-years
  segments_df = segments_df[segments_df.num_person_years >= args.py_filter]
  segments_df['total_cost'] = segments_df.avg_cost * segments_df.num_person_years

  segments_df.sort_values(by=args.sort_type, ascending=False).to_csv(
    os.path.join(args.output_path, 'disease_dyads_segmented.csv'),
    index=False, float_format='%.2f'
  )

  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--disease_dict', default='data/disease_dictionary.xls',
                      help='Dictionary of diseases/conditions to segment by')
  parser.add_argument('--py_filter', default=1, type=int,
                      help='Minimum number of person-years per population ' +
                      'segment to include in results')
  parser.add_argument('--gender_bucket', default=False, type=str2bool,
                      help='Filter by gender [True/False]')
  parser.add_argument('--age_bucket', default=False, type=str2bool,
                      help='Bucket by age group [True/False]')
  parser.add_argument('--sort_type', default='total_cost',
                      choices=['total_cost', 'avg_cost', 'num_person_years'],
                      help='Sort results by: total_cost, avg_cost, '
                      'or num_person_years]')
  parser.add_argument('--input_file', type=str, required=True,
                      help='Location of input CSV file containing claims by ' +
                      'person-year')
  parser.add_argument('--output_path', default='output',
                      help='Directory to put output files in')

  args = parser.parse_args()
  
  if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
  
  main(args)