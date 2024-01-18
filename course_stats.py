import datetime
import os
import sys

import cloudpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple, List

import pymc as pm
import arviz as az

# Our main plotting package (must have explicit import of submodules)
import bokeh.io
import bokeh.plotting

## Data and Model Code

default_data_dir = '/content/gdrive/MyDrive/CrossCountry/XCStats'

# https://discourse.pymc.io/t/how-save-pymc-v5-models/13022

def save_model(filename, model, trace, map_estimate,
               top_runner_percent, panda_data,
               course_mapper, runner_mapper,
               default_dir=default_data_dir):
  if filename.startswith('/'):
    full_filename = filename
  else:
    full_filename = os.path.join(default_dir, filename)

  dict_to_save = {'model': model,
                  'trace': trace,
                  'top_runner_percent': top_runner_percent,
                  'panda_data': panda_data,
                  'course_mapper': course_mapper,
                  'runner_mapper': runner_mapper,
                  'map_estimate': map_estimate,
                  }

  with open(full_filename , 'wb') as buff:
      cloudpickle.dump(dict_to_save, buff)


def load_model(filename,
               default_dir='/content/gdrive/MyDrive/CrossCountry/XCStats/'):
  if filename.startswith('/'):
    full_filename = filename
  else:
    full_filename = os.path.join(default_dir, filename)
  with open(full_filename , 'rb') as buff:
      model_dict = cloudpickle.load(buff)
  return (model_dict['model'], model_dict['trace'],
          model_dict['top_runner_percent'], model_dict['panda_data'],
          model_dict['course_mapper'], model_dict['runner_mapper'],
          model_dict['map_estimate'])


###Synthesize Data

def generate_xc_data(n: int = 4000,
                     num_runners: int = 400,
                     num_courses: int = 5,
                     standard_time: float = 18*60, # seconds
                     monthly_improvement: float = 10, # Seconds
                     yearly_improvement: float = 20, # Seconds
                     noise: float = 10, # seconds
                     use_month: bool = True,
                     use_year: bool = True,
                     use_course: bool = True,
                     use_runner: bool = True,
                     ):
  race_month = np.random.uniform(0, 4, n)
  runner_year = np.random.randint(0, 4, n)

  course_difficulties = np.linspace(0.75, 1.25, num_courses)
  course_ids = np.random.randint(0, len(course_difficulties), n)

  runner_abilities = np.maximum(0, 1 + np.random.randn(num_runners)/4)
  runner_ids = np.random.randint(0, num_runners, n)

  # Generate the timing ground truth data
  course_scale = course_difficulties[course_ids]
  runner_scale = runner_abilities[runner_ids]
  times = np.ones(n, dtype=float)*standard_time
  if use_year: times -= runner_year*yearly_improvement
  if use_month: times -= race_month*monthly_improvement
  if use_course: times *= course_scale
  if use_runner: times *= runner_scale
  times += np.random.randn(n)*noise

  df = pd.DataFrame(data={'race_month': race_month,
                          'runner_year': runner_year,
                          'course_ids': course_ids,
                          'runner_ids': runner_ids,
                          'times': times})
  return df, course_difficulties, runner_abilities

def create_xc_model(data: pd.DataFrame,
                    use_month: bool = True,
                    use_year: bool = True,
                    use_course: bool = True,
                    use_runner: bool = True,
                    noise_seconds: float = 10,
                    ):
  """Build a model connecting runner and course parameters,
  along with monthly and yearly improvements to race time. This model
  assumes the following fields are in the Panda dataframe.
    race_month (usually 0-3 for the fall months)
    runner_year (usually 0-3 for the runners 4 years of HS
    course_ids (which course is run, a small integer)
    runner_ids (a small integer)
  """
  mean_time = np.mean(data.times.values)
  num_courses = len(set(data.course_ids.values))
  num_runners = len(set(data.runner_ids.values))
  print(f'Building a XC model for {num_runners} runners '
        f'running {num_courses} courses')
  # https://twiecki.io/blog/2014/03/17/bayesian-glms-3/
  with pm.Model() as a_model:
    # Intercept prior
    bias = pm.Normal('bias', mu=mean_time, sigma=100)
    # Month prior
    if use_month:
      monthly_slope = pm.Normal('monthly_slope', mu=0, sigma=100)

    # Year prior
    if use_year:
      yearly_slope = pm.Normal('yearly_slope', mu=0, sigma=100)

    # Course IDs
    if use_course:
      course_est = pm.Normal('course_est', mu=1, sigma=1,
                             shape=num_courses)

    if use_runner:
      runner_est = pm.Normal('runner_est', mu=1, sigma=1,
                             shape=num_runners)
    # Model error prior
    eps = noise_seconds*pm.HalfCauchy('eps', beta=1)

    # Now put it all together to match the time predictions.
    time_est = bias
    if use_month:
      time_est -= monthly_slope * data.race_month.values
    if use_year:
      time_est -= yearly_slope * data.runner_year.values
    if use_course:
      time_est *= course_est[data.course_ids.values]
    if use_runner:
      time_est *= runner_est[data.runner_ids.values]
    # time_est += eps

    # Data likelihood
    y_like = pm.Normal('y_like', mu=time_est, sigma=eps,
                       observed=data.times.values)
  return a_model


#######  XCStats Import and Transforms ######

def parse_date(date_string: str) -> datetime.datetime:
  """Parse the XCStats date format."""
  return datetime.datetime.strptime(date_string, '%m/%d/%Y')

def extract_month(date, starting_month=0):
  """Get the race month as a number.  Starting_month allows us to start counting
  from September (9).  Return an integer (generally between 0 and 3)
  """
  return parse_date(date).month - starting_month

def extract_year(date):
  """Return the year of the race as an integer."""
  return parse_date(date).year

def import_xcstats(
    csv_file: str,
    default_dir: str = default_data_dir) -> pd.DataFrame:
  """Import the XCStats data into Panda.  Transform the race date into the
  month and year we need for our model.  The month is number of months since
  September, since that's the start of the season (generally 0-3).  The year is
  the runner's year in high school (0-3).  Divide the times and distances by
  100 to turn into seconds and miles.

  Return a new Panda DataFrame.
  """
  if csv_file.startswith('/'):
    filename = csv_file
  else:
    filename = os.path.join(default_dir, csv_file)
  data = pd.read_csv(filename)

  # Adjust the dates and put them in the right format
  months = data.meetDate.apply(extract_month, starting_month=9)
  years = data.meetDate.apply(extract_year)

  data['times'] = data.result / 100.0
  data['distance'] = data.distance / 100.0

  # Add mileage to each course name since sometimes the same course has multiple
  # distances.
  # https://saturncloud.io/blog/how-to-combine-two-columns-in-a-pandas-dataframe/
  def race_name(row):
    return f'{row["courseName"]} ({row["distance"]})'
  data['course_distance'] = data.apply(race_name, axis=1)

  race_data = pd.DataFrame({'race_month': months,
                            'race_year': years})
  data_with_dates = pd.concat((data, race_data), axis=1)
  return data_with_dates

def transform_ids(data: pd.DataFrame,
                  column_name: str) -> Tuple[pd.Series, Dict[Any, int]]:
  """Go through all the indicated column data and generate a mapping from
  the original name/id to a small number.  Return the time series, and
  the dictionary that maps the original name/id into the new ID.
  """
  original_ids = list(set(data[column_name]))
  mapping_dictionary = dict(zip(original_ids, range(len(original_ids))))
  # print(mapping_dictionary)
  return data[column_name].map(mapping_dictionary), mapping_dictionary

def prepare_xc_data(data: pd.DataFrame,
                    school_id: Optional[int] = None,
                    place_fraction: Optional[float] = None,
                    remove_bad_grads: bool = True
                    ) -> Tuple[pd.DataFrame,
                               Dict[Any, int],
                               Dict[Any, int]]:
  """Put the XCStats' data into the right Panda form for analysis my the
  Bayesian model code.  In particular, this creates the umique runner_ids
  and course_ids fields, by renumbering them so they are consequitive. Return
  a new dataframe with all the original fields, and the ones we add here.

  The school_id argument limits the output to just that school.
  The place_fraction limits the runners to those that finish in this *fraction*
  of the top.
  """
  if school_id:
    data = data[data['schoolID'] == school_id].copy()

  if place_fraction:
    data = data[data['place']/data['num_runners'] < place_fraction].copy()

  if remove_bad_grads:
    # Some runners are missing a graduation year (set to zero) so remove them.
    data = data[data['gradYear'] > 1900].copy()

  new_runner_ids, runner_mapper = transform_ids(data, 'runnerID')
  data.loc[:, 'runner_ids'] = new_runner_ids
  new_course_ids, course_mapper = transform_ids(data, 'course_distance')
  data.loc[:, 'course_ids'] = new_course_ids

  # Years since a freshman (which will be 0)
  data.loc[:, 'runner_year'] = data['race_year'] - (data['gradYear'] - 4)
  return data, runner_mapper, course_mapper

  
def build_and_test_model(xc_data: pd.DataFrame, chains=2) -> Tuple[
    pm.Model, dict[str, np.ndarray], az.InferenceData]:
  """Find the MAP and parameter distributions for the given data."""
  xc_model = create_xc_model(xc_data)
  print(f'Find the MAP estimate for {xc_data.shape[0]} results....')
  map_estimate = pm.find_MAP(model=xc_model)

  print(f'Find the MCMC distribution for {xc_data.shape[0]} results....')
  model_trace = pm.sample(model=xc_model, chains=chains)
  return xc_model, map_estimate, model_trace


def find_course_name(name, mapper):
  return [k for k in mapper.keys() if name in k]


def save_model(filename, model, trace, map_estimate,
               top_runner_percent, panda_data,
               course_mapper, runner_mapper,
               default_dir=default_data_dir):
  if filename.startswith('/'):
    full_filename = filename
  else:
    full_filename = os.path.join(default_dir, filename)

  dict_to_save = {'model': model,
                  'trace': trace,
                  'top_runner_percent': top_runner_percent,
                  'panda_data': panda_data,
                  'course_mapper': course_mapper,
                  'runner_mapper': runner_mapper,
                  'map_estimate': map_estimate,
                  }

  with open(full_filename , 'wb') as buff:
      cloudpickle.dump(dict_to_save, buff)


def load_model(filename,
               default_dir='/content/gdrive/MyDrive/CrossCountry/XCStats/'):
  if filename.startswith('/'):
    full_filename = filename
  else:
    full_filename = os.path.join(default_dir, filename)
  with open(full_filename , 'rb') as buff:
      model_dict = cloudpickle.load(buff)
  return (model_dict['model'], model_dict['trace'],
          model_dict['top_runner_percent'], model_dict['panda_data'],
          model_dict['course_mapper'], model_dict['runner_mapper'],
          model_dict['map_estimate'])


def create_markdown_table(df):
  print('|Index | Course Name                      | Boys Difficulty | '
        'Girls Difficulty | # Boys | # Girls |')
  print('|-----:|---------------------------------:|'
        '----------------:|'
        '-----------------:|-------:|--------:|')
  for index, row in df.iterrows():
    print(f'|{index:5d} | {row["course_name"]:32s} | '
          f'{row["vb_difficulty"]:2.3f}           |'
          f'{row["vg_difficulty"]:2.3f}             |'
          f'{row["boys_runner_count"]:5d}   |',
          f'{row["girls_runner_count"]:5d}   |')


html_header = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>California Course Difficulties</title>
        <link rel="stylesheet" href="https://www.w3.org/WAI/content-assets/wai-aria-practices/patterns/table/examples/css/sortable-table.css">
        <script src="https://www.w3.org/WAI/content-assets/wai-aria-practices/patterns/table/examples/js/sortable-table.js"></script>
    </head>
    <body>
    <!--https://www.w3.org/WAI/ARIA/apg/patterns/table/examples/sortable-table/-->
        <div class="table-wrap">
            <table class="sortable">
                <caption>
                    Course difficulities by Bayesian Modeling
                    <span class="sr-only">, column headers with buttons are sortable.</span>
                </caption>
                <thead>
                    <tr>
                        <th aria-sort="ascending">
							            <button>Course Name<span aria-hidden="true"></span></button>
						            </th>
                        <th class="num">
							            <button>Boys Difficulty<span aria-hidden="true"></span></button>
						            </th>
                        <th class="num">
							            <button>Girls Difficulty<span aria-hidden="true"></span></button>
						            </th>
                        <th class="num">
                          <button># Boys<span aria-hidden="true"></span></button>
						            </th>
                        <th class="num">
							            <button># Girls<span aria-hidden="true"></span></button>
						            </th>
                        <th>Local</th>
                    </tr>
                </thead>
                <tbody>
"""

html_footer = """
      </table>
    </div>
    <p>       
  </body>
</html>
"""

# Note, for reasons I don't understand, at least one field in the table headers
# must be non sortable.  I'm using local for that now.

def create_html_table(df, filename):
  with open(filename, 'w') as f:
    print(html_header, file=f)
    for index, row in df.iterrows():
      print('<tr>', file=f)
      print(f'<td>{row["course_name"]}</td>', 
            f'<td>{row["vb_difficulty"]:2.4f}</td>'
            f'<td>{row["vg_difficulty"]:2.4f}</td>'
            f'<td>{row["boys_runner_count"]:5d}</td>'
            f'<td>{row["girls_runner_count"]:5d}</td>'
            f'<td>{row["local_course"]}</td>'
            f'</tr>', file=f)
    print(html_footer, file=f)


def plot_difficulty_comparison(scatter_df:
  # Drop the data point for Spooner since it's a very long distance.
  p = bokeh.plotting.figure(title="Varsity Boy/Girl Course Difficulty Comparison",
                            x_axis_label='Girls MAP Estimate',
                            y_axis_label='Boys MAP Estimate',
                            x_range=(0.2, 1.4),  # Skip Spooner
                            y_range=(0.2, 1.4),  # Skip Spooner
                            tooltips=[
                                ("Course Name", "@course_name"),
                                ("Course Distance", "@course_distances"),
                                ("Course Difficulty (Boys)", "@vb_difficulty"),
                                ("Course Difficulty (Girls)", "@vg_difficulty"),
                                ("Boys Runner Count", "@boys_runner_count"),
                                ("Girls Runner Count", "@girls_runner_count")
                                ])

  # Compare the boys and girls course difficulties with a scatter plot
  far_data = scatter_df.loc[~scatter_df['local_course']]
  p.cross(source=far_data, x='vg_difficulty', y='vb_difficulty',
          legend_label="Other Courses",
          line_color='blue')

  local_data = scatter_df.loc[scatter_df['local_course']]
  p.circle(source=local_data, x='vg_difficulty', y='vb_difficulty',
          legend_label="Local Courses",
          line_color='red', fill_color='red', size=6)

  p.legend.location = 'top_left'
  return p

main(argv)

if __name__ == '__main__':
  main(sys.argv)