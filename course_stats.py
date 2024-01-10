import datetime
import os

import cloudpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple, List

import pymc as pm
import arviz as az

## Data and Model Code

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
  if use_year: 
    times -= runner_year*yearly_improvement
  if use_month: 
    times -= race_month*monthly_improvement
  if use_course: 
    times *= course_scale
  if use_runner: 
    times *= runner_scale
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

parse_date('10/5/2019')

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
    default_dir: str ='/content/gdrive/MyDrive/CrossCountry/XCStats') -> pd.DataFrame:
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
  new_course_ids, course_mapper = transform_ids(data, 'courseName')
  data.loc[:, 'course_ids'] = new_course_ids

  data.loc[:, 'runner_year'] = data['gradYear'] - data['race_year']
  return data, runner_mapper, course_mapper

  
def build_and_test_model(xc_data: pd.DataFrame) -> Tuple[
    pm.Model, dict[str, np.ndarray], az.InferenceData]:
  """Find the MAP and parameter distributions for the given data."""
  xc_model = create_xc_model(xc_data)
  print(f'Find the MAP estimate for {xc_data.shape[0]} results....')
  map_estimate = pm.find_MAP(model=xc_model)

  print(f'Find the MCMC distribution for {xc_data.shape[0]} results....')
  model_trace = pm.sample(model=xc_model)
  return xc_model, map_estimate, model_trace


def save_model(filename, model, trace, map_estimate, 
               top_runner_percent, panda_data, 
               course_mapper, runner_mapper,
               default_dir='/content/gdrive/MyDrive/CrossCountry/XCStats/'):
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
