"""Compute the parameters of a Bayesian model that predicts school race times.

Note: The XCStats IDs are not necessarily consecutive, so we transform their ID
into an index, which are consequtive indices and good for the Bayesian model
software.
"""
import datetime
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import app
from absl import flags

import cloudpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc as pm
import pytensor.tensor as pt
import arviz as az

# Our main plotting package (must have explicit import of submodules)
import bokeh.io
import bokeh.plotting

######################## Data and Model Code ##############################

DEFAULT_DATA_DIR = 'Data'
DEFAULT_CACHE_DIR = 'Cache'

# https://discourse.pymc.io/t/how-save-pymc-v5-models/13022


# pylint: disable=too-many-arguments # Do this globally, funtions are complex.
# pylint: disable=too-many-locals
def save_model(filename, model, trace, map_estimate,
               top_runner_percent, panda_data,
               course_id_to_index, runner_id_to_index, course_course_id_to_name,
               default_dir: Optional[str] = DEFAULT_CACHE_DIR) -> None:
  if filename.startswith('/') or default_dir is None:
    full_filename = filename
  else:
    full_filename = os.path.join(default_dir, filename)

  dict_to_save = {'model': model,
                  'trace': trace,
                  'top_runner_percent': top_runner_percent,
                  'panda_data': panda_data,
                  'course_id_to_index': course_id_to_index,
                  'runner_id_to_index': runner_id_to_index,
                  'course_course_id_to_name': course_course_id_to_name,
                  'map_estimate': map_estimate,
                  'datetime':
                      datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                  }

  with open(full_filename, 'wb') as buff:
    cloudpickle.dump(dict_to_save, buff)


def load_model(
    filename,
    default_dir: Optional[str] = DEFAULT_CACHE_DIR) -> Tuple[
      pm.Model, az.InferenceData, Any, pd.DataFrame, Dict, Dict, Dict, Dict
    ]:
  if filename.startswith('/') or default_dir is None:
    full_filename = filename
  else:
    full_filename = os.path.join(default_dir, filename)
  with open(full_filename, 'rb') as buff:
    model_dict = cloudpickle.load(buff)
    if datetime in model_dict:
      print(f'Restoring data stored at {model_dict["datetime"]}')
  return (model_dict['model'], model_dict['trace'],
          model_dict['top_runner_percent'], model_dict['panda_data'],
          model_dict['course_id_to_index'], model_dict['runner_id_to_index'],
          model_dict['course_course_id_to_name'],
          model_dict['map_estimate'])


######################## Synthesize Data ##############################

def generate_xc_data(n_samples: int = 4000,
                     num_runners: int = 800,
                     num_courses: int = 5,
                     standard_time: float = 18*60,  # seconds
                     monthly_improvement: float = 10,  # Seconds
                     yearly_improvement: float = 20,  # Seconds
                     noise: float = 10,  # seconds
                     use_month: bool = True,  # Include monthly improvement
                     use_year: bool = True,  # Include year-by-year improvement
                     use_course: bool = True,  # Vary course difficulties
                     use_runner: bool = True,  # Vary runner slowness
                     ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
  """Generate some synthetic data to test our models.

  Returns:
    a tuple consisting of
    1) a Panda data frame,
    2) the underlying course difficulties (num_courses entries) and
    3) the underlying runner abilities (num_runners entries)
  """
  race_month = np.random.uniform(0, 4, n_samples)
  runner_year = np.random.randint(0, 4, n_samples)

  course_difficulties = np.linspace(0.75, 1.25, num_courses)
  course_ids = np.random.randint(0, len(course_difficulties), n_samples)

  runner_abilities = np.maximum(0, 1 + np.random.randn(num_runners)/4)
  runner_ids = np.random.randint(0, num_runners, n_samples)

  # Generate the timing ground truth data
  course_scale = course_difficulties[course_ids]
  runner_scale = runner_abilities[runner_ids]
  times = np.ones(n_samples, dtype=float)*standard_time
  if use_year:
    times -= runner_year*yearly_improvement
  if use_month:
    times -= race_month*monthly_improvement
  if use_course:
    times *= course_scale
  if use_runner:
    times *= runner_scale
  times += np.random.randn(n_samples)*noise

  new_df = pd.DataFrame(data={'race_month': race_month,
                              'runner_year': runner_year,
                              'course_ids': course_ids,
                              'runner_ids': runner_ids,
                              'times': times})
  return new_df, course_difficulties, runner_abilities


def create_xc_model(data: pd.DataFrame,  # pylint: disable=too-many-locals
                    month_spec: Optional[str] = 'normal,0,100',
                    year_spec: Optional[str] = 'normal,0,100',
                    course_spec: Optional[str] = 'normal,1,1',
                    runner_spec: Optional[str] = 'normal,1,1',
                    noise_seconds: float = 10,
                    ) -> pm.Model:
  """Build a model connecting runner and course parameters,
  along with monthly and yearly improvements to race time. This model
  assumes the following fields are in the Panda dataframe.
    race_month (usually 0-3 for the fall months)
    runner_year (usually 0-3 for the runners 4 years of HS
    course_ids (which course is run, a small integer)
    runner_ids (a small integer)
  """
  def make_model(name: str,
                 shape: Union[int, Tuple[int]],
                 definition: Optional[str]) -> Union[float, pm.Distribution]:
    """Given a distribution spec, create the PYMC object that implements it.

    Args:
      name: The name of the variable with the given distribution.
        Use None for skipping this distribution and returning a constant 1.0
      shape: The shape of the distribution, either an integer or a tuple
      definition: A distribution name, and optional parameters, comma separated.
        e.g. normal,0,1 means create a Normal distribution with mean 0 and
        sigma 1.

    Returns:
      Either 1.0 (for none or constant variables) or a PYMC distribution object.
    """
    if not definition:
      print(f'Skipping the model for {name}.')
      return 1.0
    specs = definition.split(',')
    print(f'Creating a {name} distribution for {definition} with size {shape}')
    if specs[0].lower() == 'constant' or specs[0].lower() == 'none':
      return 1.0
    elif specs[0].lower() == 'normal':
      if len(specs) > 1:
        mean = float(specs[1])
      else:
        mean = 0.0
      if len(specs) > 2:
        sigma = float(specs[2])
      else:
        sigma = 1
      return pm.Normal(name, mu=mean, sigma=sigma, shape=shape)
    elif specs[0].lower() == 'gamma':
      # https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Gamma.html#pymc.Gamma
      if len(specs) > 1:
        mean = float(specs[1])
      else:
        mean = 0.0
      if len(specs) > 2:
        sigma = float(specs[2])
      else:
        sigma = 1
      assert sigma > 0, 'Sigma in a Gamma distribution must be greater than 0'
      return pm.Gamma(name, mu=mean, sigma=sigma, shape=shape)
    else:
      raise ValueError(f'Unknown model type: {definition}')

  mean_time = np.mean(data.times.values)
  num_courses = int(max(data.course_ids.values)) + 1
  num_runners = int(max(data.runner_ids.values)) + 1
  print(f'Building a XC model for {num_runners} runners '
        f'running {num_courses} courses')
  # https://twiecki.io/blog/2014/03/17/bayesian-glms-3/
  with pm.Model() as a_model:
    # Intercept prior
    sigma_time = 100
    print(f'Creating a N({mean_time}, {sigma_time}) distribution for the bias.')
    bias = pm.Normal('bias', mu=mean_time, sigma=sigma_time)
    monthly_slope = make_model('monthly_slope', 1, month_spec)
    yearly_slope = make_model('yearly_slope', 1, year_spec)
    course_est = make_model('course_est', num_courses, course_spec)
    runner_est = make_model('runner_est', num_runners, runner_spec)

    # Model error prior
    eps = noise_seconds*pm.HalfCauchy('eps', beta=1)

    # Now put it all together to match the time predictions.
    time_est = bias
    if isinstance(monthly_slope, pt.variable.TensorVariable):
      time_est -= monthly_slope * data.race_month.values
    if isinstance(yearly_slope, pt.variable.TensorVariable):
      time_est -= yearly_slope * data.runner_year.values
    if isinstance(course_est, pt.variable.TensorVariable):
      time_est *= course_est[data.course_ids.values]  # pylint: disable=unsubscriptable-object
    if isinstance(runner_est, pt.variable.TensorVariable):
      time_est *= runner_est[data.runner_ids.values]  # pylint: disable=unsubscriptable-object

    # Data likelihood
    pm.Normal('y_like', mu=time_est, sigma=eps, observed=data.times.values)
  return a_model


#######  XCStats Import and Transforms ######

def parse_date(date_string: str) -> datetime.datetime:
  """Parse the XCStats date format."""
  try:
    return datetime.datetime.strptime(date_string, '%m/%d/%Y')
  except ValueError:
    # Now look for a two-digit year.
    return datetime.datetime.strptime(date_string, '%m/%d/%y')


def extract_month(date, starting_month=0):
  """Get the race month as a number.  Starting_month allows us to start counting
  from September (9).  Return an integer (generally between 0 and 3)
  """
  return parse_date(date).month - starting_month


def extract_year(date):
  """Return the year of the race as an integer."""
  return parse_date(date).year


# The CSV has these fields:
#  meetDate	runnerID	gradYear	gender	schoolID	result	division	courseID	
#   courseName distance	place	 num_runners state

def import_xcstats(
    csv_file: str,
    default_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
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
  # def race_name(row):
  #   return f'{row["courseName"]} ({row["distance"]})'
  # data['course_distance'] = data.apply(race_name, axis=1)

  race_data = pd.DataFrame({'race_month': months,
                            'race_year': years})
  data_with_dates = pd.concat((data, race_data), axis=1)
  return data_with_dates


def find_course_course_id_to_names(data: pd.DataFrame) -> Dict[int, str]:
  """Create dictionary mapping XCStats ID to my course_name_distance string.
  """
  assert isinstance(data, pd.DataFrame), f'data is a {type(data)}'
  names = {}
  for _, a_result in data.iterrows():
    id = a_result.courseID
    if id not in names:
      names[id] = f'{a_result["courseName"]} ({a_result["distance"]})'
  return names


def transform_ids(data: pd.DataFrame,
                  column_name: str) -> Tuple[pd.Series, Dict[Any, int]]:
  """Go through all the indicated column data and generate a mapping from
  the original name/id to a small number.  Return the time series, and
  the dictionary that maps the original name/id into the new indices.
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

  Returns:
    The new Panda dataframe, with added fields for runner_ids and course_ids,
    which are now consequitive integers.  Also returns dictionaries that map 
    the original ID to the model index (a set of consecutive indices).
  """
  if school_id:
    data = data[data['schoolID'] == school_id].copy()

  if place_fraction:
    data = data[data['place']/data['num_runners'] < place_fraction].copy()

  if remove_bad_grads:
    # Some runners are missing a graduation year (set to zero) so remove them.
    data = data[data['gradYear'] > 1900].copy()

  new_runner_ids, runner_id_to_index = transform_ids(data, 'runnerID')
  data.loc[:, 'runner_ids'] = new_runner_ids
  # new_course_ids, course_id_to_index = transform_ids(data, 'course_distance')
  new_course_ids, course_id_to_index = transform_ids(data, 'courseID')
  data.loc[:, 'course_ids'] = new_course_ids

  # Years since a freshman (which will be 0)
  data.loc[:, 'runner_year'] = data['race_year'] - (data['gradYear'] - 4)
  return data, runner_id_to_index, course_id_to_index


def build_and_test_model(xc_data: pd.DataFrame,
                         monthly_spec: str,
                         yearly_spec: str,
                         course_spec: str,
                         runner_spec: str,
                         chains: int = 2,
                         draws: int = 1000,
                         tune: int = 1000,  # Shorten for debugging
                         seed: Optional[np.random.Generator] = None) -> Tuple[
    pm.Model, dict[str, np.ndarray], az.InferenceData]:
  """Find the MAP and parameter distributions for the given data."""
  xc_model = create_xc_model(xc_data, monthly_spec, yearly_spec,
                             course_spec, runner_spec)
  print(f'Find the MAP estimate for {xc_data.shape[0]} results.')
  print(f'   Calculating {chains} chains each with {draws} draws.')
  map_estimate = pm.find_MAP(model=xc_model)

  print(f'Find the MCMC distribution for {xc_data.shape[0]} results....')
  model_trace = pm.sample(model=xc_model, chains=chains,
                          draws=draws, tune=tune, random_seed=seed)
  return xc_model, map_estimate, model_trace


def find_course_name(name, mapper):
  return [k for k in mapper.keys() if name in k]


# Gather the data so we can plot a scatter plot showing boy's and girl's
# course difficuties.
def create_result_frame(
    vb_data: pd.DataFrame,
    vg_data: pd.DataFrame,
    vb_course_id_to_index: Dict[Any, int],
    vg_course_id_to_index: Dict[Any, int],
    vb_course_id_to_name: Dict[int, str],
    vb_model_trace: az.InferenceData, 
    vg_model_trace: az.InferenceData,
    local_course_list=(),
    vb_map_estimate=None, vg_map_estimate=None,
    use_map=False, normalize_to_crystal=True):
  if use_map:
    vb_course_est = vb_map_estimate['course_est']
    vg_course_est = vg_map_estimate['course_est']
  else:
    # Average the posterior for course_est over all traces and all samples.
    # The result is an NDArray in the order of the consecutive course indices.
    vb_course_est = np.mean(vb_model_trace.posterior.course_est.values,
                            axis=(0, 1))
    vg_course_est = np.mean(vg_model_trace.posterior.course_est.values,
                            axis=(0, 1))

  # Find XCStats course ids that are common to both the boys and girls races.
  common_courses = find_common_courses(vb_course_id_to_index, vg_course_id_to_index)

  vg_difficulties = []
  vb_difficulties = []
  course_distances = []
  boys_runner_count = []
  girls_runner_count = []
  local_course = []
  for course_id in common_courses:  # These are the XCStats CourseIDs
    course_name = vb_course_id_to_name[course_id]
    base_course_name, distance = get_course_distance(course_name)
    course_distances.append(distance)

    vb_difficulties.append(vb_course_est[vb_course_id_to_index[course_id]])
    vg_difficulties.append(vg_course_est[vg_course_id_to_index[course_id]])
    if course_name == 'Crystal Springs (2.95)':
      vb_norm = vb_difficulties[-1]
      vg_norm = vg_difficulties[-1]

    runner_ids = vb_data.loc[
      vb_data['courseID'] == course_id]['runnerID'].values
    boys_runner_count.append(len(runner_ids))
    runner_ids = vg_data.loc[
      vg_data['courseID'] == course_id]['runnerID'].values
    girls_runner_count.append(len(runner_ids))

    local_course.append(base_course_name in local_course_list)

  if normalize_to_crystal:
    vb_difficulties = np.asarray(vb_difficulties)/vb_norm
    vg_difficulties = np.asarray(vg_difficulties)/vg_norm

  scatter_df = pd.DataFrame({'vg_difficulty': vg_difficulties,
                             'vb_difficulty': vb_difficulties,
                             'course_name': common_courses,
                             'course_distances': course_distances,
                             'boys_runner_count': boys_runner_count,
                             'girls_runner_count': girls_runner_count,
                             'local_course': local_course,
                            })
  return scatter_df


def create_markdown_table(race_data: pd.DataFrame):
  print('In create_markdown_table:')
  print('Column names are:', race_data.columns.tolist())
  print(race_data.head().to_string())
  print('|Index | Course Name                      | Boys Difficulty | '
        'Girls Difficulty | # Boys | # Girls |')
  print('|-----:|---------------------------------:|'
        '----------------:|'
        '-----------------:|-------:|--------:|')
  for index, row in race_data.iterrows():
    print(f'|{index:5d} | {row["course_name"]:32s} | '
          f'{row["vb_difficulty"]:2.3f}           |'
          f'{row["vg_difficulty"]:2.3f}             |'
          f'{row["boys_runner_count"]:5d}   |',
          f'{row["girls_runner_count"]:5d}   |')


HTML_HEADER = """
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
                    <b>Course difficulities by Bayesian Modeling (Click on column hearing to sort)</b>
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
                    </tr>
                </thead>
                <tbody>
"""

HTML_FOOTER = """
      </table>
    </div>
    <p>
    <style>table.sortable th button { position: relative; }</style>
  </body>
</html>
"""

# Note, for reasons I don't understand, at least one field in the table headers
# must be non sortable.  I'm using local for that now.


def create_html_table(race_data: pd.DataFrame,
                      filename: str,
                      title: str = None):
  with open(filename, 'w', encoding="utf-8") as file_pointer:
    if title:
      my_header = HTML_HEADER.replace('California Course Difficulties', title)
    else:
      my_header = HTML_HEADER
    print(my_header, file=file_pointer)
    for _, row in race_data.iterrows():
      print('<tr>', file=file_pointer)
      print(f'<td>{row["course_name"]}</td>',
            f'<td class="num">{row["vb_difficulty"]:2.4f}</td>'
            f'<td class="num">{row["vg_difficulty"]:2.4f}</td>'
            f'<td class="num">{row["boys_runner_count"]}</td>'
            f'<td class="num">{row["girls_runner_count"]}</td>'
            f'</tr>', file=file_pointer)
    print(HTML_FOOTER, file=file_pointer)


# Figure out which courses are common to both boys and girls (for the scatter
# that follow.)
def find_common_courses(vb_course_id_to_index: Dict[int, int], 
                        vg_course_id_to_index: Dict[int, int]) -> List[int]:
  """Find the courses that are common to both boys and girls.  The mappers
  map from original XCStats Course ID to the small integer that we use when
  building the model.
  """
  vb_courses = set(vb_course_id_to_index.keys())
  vg_courses = set(vg_course_id_to_index.keys())
  return list(set(vb_courses.intersection(vg_courses)))


# Extract the course names where our local schools run, for easier debugging.
LOCAL_SCHOOLS = (830,  # Palo Alto
                 950,  # Los Altos
                 1,  # Archbishop Mitty
                 10,  # Lynbrook
)


def find_local_courses(pd_data: pd.DataFrame,
                       local_schools: List[int] = LOCAL_SCHOOLS):
  local_courses = pd_data.loc[pd_data['schoolID'].isin(local_schools)]
  return local_courses.courseName.unique()


def get_course_distance(name_and_dist: str) -> str:
  """Parse the course-name distance string into its two pieces."""
  pieces = name_and_dist.rsplit(' ', 1)
  return pieces[0], float(pieces[1][1:-1])


def create_hank_correction_list(race_data: pd.DataFrame, filename: str):
  """Create the HTML table that Hank Lawson needs for the Lynbrook course
  timing calculator.
    https://lynbrooksports.prepcaltrack.com/ATHLETICS/XC/CONVERTR/converter2007.html
  """
  with open(filename, 'w', encoding="utf-8") as file_pointer:
    for _, row in race_data.iterrows():
      difficulty = (row["vb_difficulty"] + row["vg_difficulty"])/2.0
      print(f'<option value="{difficulty:2.4f}">{row["course_name"]}</option>',
            file=file_pointer)


################## Plotting Routines ########################

def plot_map_course_difficulties(
    map_estimates,
    title: str = 'Histogram of Course Difficulties (MAP)',
    filename: str = None,
    difficulty_limit: int = 3):
  course_data = map_estimates['course_est']
  course_data[course_data > difficulty_limit] = np.nan  # Drop Spooner
  plt.clf()
  plt.hist(course_data, 20)
  plt.title(title)
  plt.xlabel('Course Difficulty (arbitrary units)')
  if filename:
    plt.savefig(filename)


def plot_map_runner_abilities(map_estimates,
                              title='Histogram of Runner Abilities (MAP)',
                              filename=None,
                              difficulty_limit=3):
  course_data = map_estimates['runner_est']
  course_data[course_data > difficulty_limit] = np.nan  # Drop Spooner
  plt.clf()
  plt.hist(course_data, 20)
  plt.title(title)
  plt.xlabel('Relative Runner Abilities (arbitrary units)')
  if filename:
    plt.savefig(filename)


def line_hist(data, bins=10, **kwargs):
  y_locs, bin_edges = np.histogram(data, bins=bins)
  bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
  plt.plot(bincenters, y_locs, **kwargs)


def plot_monthly_slope_predictions(
    trace_data: az.InferenceData,
    title: str = 'Histogram of Monthly Slope Predictions',
    num_bins: int = 20,
    filename: Optional[str] = None) -> Optional[List[np.ndarray]]:

  def add_line(x_loc, label):
    axis_limits = plt.axis()
    plt.plot([x_loc, x_loc], axis_limits[2:], '--', label=label)
    plt.text(x_loc, np.mean(axis_limits[2:]), f'{x_loc:4.3}')

  if 'monthly_slope' not in trace_data.posterior:
    return None
  monthly_trace_means = []
  slopes = trace_data.posterior.monthly_slope.values
  min_slope = np.min(slopes.flatten())
  max_slope = np.max(slopes.flatten())
  bins = np.linspace(min_slope, max_slope, num_bins+1)
  plt.clf()
  for i in range(slopes.shape[0]):
    monthly_trace_means.append(np.mean(slopes[i, :]))
    line_hist(slopes[i, :], bins=bins, alpha=0.5, label=f'Trace {i}')
  plt.xlabel('Monthly Improvement During Each Season (s)')
  plt.title(title)

  add_line(np.mean(slopes), 'Trace Mean')
  if slopes.shape[0] < 8:
    plt.legend()

  if filename:
    plt.savefig(filename)
  return monthly_trace_means


def plot_yearly_slope_predictions(
    trace_data: az.InferenceData,
    title: str = 'Histogram of Yearly Slope Predictions',
    num_bins: int = 20,
    filename: Optional[str] = None) -> Optional[List[np.ndarray]]:

  def add_line(x_loc, label):
    axis_limits = plt.axis()
    plt.plot([x_loc, x_loc], axis_limits[2:], '--', label=label)
    plt.text(x_loc, np.mean(axis_limits[2:]), f'{x_loc:4.3}')

  if 'yearly_slope' not in trace_data.posterior:
    return None
  yearly_trace_means = []

  slopes = trace_data.posterior.yearly_slope.values
  min_slope = np.min(slopes.flatten())
  max_slope = np.max(slopes.flatten())
  bin_locs = np.linspace(min_slope, max_slope, num_bins+1)
  plt.clf()
  for i in range(slopes.shape[0]):
    yearly_trace_means.append(np.mean(slopes[i, :]))
    line_hist(slopes[i, :], bins=bin_locs, alpha=0.5, label=f'Trace {i}')
  plt.xlabel('Yearly Improvement During Each Season (s)')
  plt.title(title)

  add_line(np.mean(slopes), 'Trace Mean')
  if slopes.shape[0] < 8:
    plt.legend()

  if filename:
    plt.savefig(filename)
  return yearly_trace_means


def plot_map_trace_difficulty_comparison(
    map_estimate,
    model_trace: az.InferenceData,
    title: str = 'Comparison of Course Difficult Estimates',
    difficulty_limit: float = 2.2,  # Drop Spooner since it's way long.
    filename: Optional[str] = None) -> List[np.ndarray]:
  if 'course_est' not in model_trace.posterior:
    return
  # Plot the MAP vs. mean trace estimate of the course difficulties.
  course_difficulties = model_trace.posterior.course_est.values
  plt.clf()
  difficulty_trace_slopes = []
  for i in range(course_difficulties.shape[0]):
    trace_difficulties = np.mean(course_difficulties[i, :, :], axis=0)

    # Estimate the MAP vs. Trace slope
    x_data = np.vstack([map_estimate['course_est'],
                        np.ones(len(map_estimate['course_est']))])
    good_data = ~np.isnan(map_estimate['course_est'])
    line_params = np.linalg.lstsq(x_data.T[good_data, :],
                                  trace_difficulties[good_data], rcond=None)[0]
    difficulty_trace_slopes.append(line_params[0])

    plt.scatter(map_estimate['course_est'], trace_difficulties,
                label=f'Trace {i}', alpha=0.1)
  plt.xlim(0, difficulty_limit)
  plt.ylim(0, difficulty_limit)
  plt.plot([0, 2], [0, 2], '--')
  plt.xlabel('MAP Estimate')
  plt.ylabel('Mean of Trace')
  if course_difficulties.shape[0] <= 4:  # Don't bother with legend if too many
    plt.legend()
  plt.title(title)
  if filename:
    plt.savefig(filename)
  return difficulty_trace_slopes


def plot_year_month_difficulty_tradeoff(
    monthly_trace_means,
    yearly_trace_means,
    difficulty_trace_slopes,
    title='Tradeoff between year/month and course difficulties',
    filename: Optional[str] = None):
  plt.clf()
  plt.plot(monthly_trace_means,
           difficulty_trace_slopes,
          'x', label='Monthly Slope')
  plt.plot(yearly_trace_means,
           difficulty_trace_slopes,
          'o', label='Yearly Slope')
  plt.xlabel('Month or Yearly Slope')
  plt.ylabel('Mean slope of Course Difficulty')
  plt.title(title)
  plt.legend()
  if filename:
    plt.savefig(filename)


def plot_difficulty_comparison(scatter_df: pd.DataFrame,
                               filename: Optional[str] = None):
  # Drop the data point for Spooner since it's a very long distance.
  if filename:
    bokeh.plotting.output_file(filename=filename,
                               title='Difficulty Scatter Plot')
  my_plot = bokeh.plotting.figure(
    title="Varsity Boy/Girl Course Difficulty Comparison",
    x_axis_label='Girls MAP Estimate',
    y_axis_label='Boys MAP Estimate',
    x_range=(0.2, 1.4),  # Skip Spooner
    y_range=(0.2, 1.4),  # Skip Spooner
    tooltips=[("Course Name", "@course_name"),
              ("Course Distance", "@course_distances"),
              ("Course Difficulty (Boys)", "@vb_difficulty"),
                                ("Course Difficulty (Girls)", "@vg_difficulty"),
              ("Boys Runner Count", "@boys_runner_count"),
              ("Girls Runner Count", "@girls_runner_count")
              ])

  # Compare the boys and girls course difficulties with a scatter plot
  far_data = scatter_df.loc[~scatter_df['local_course']]
  my_plot.cross(source=far_data, x='vg_difficulty', y='vb_difficulty',
                legend_label="Other Courses",
                line_color='blue')

  local_data = scatter_df.loc[scatter_df['local_course']]
  my_plot.circle(source=local_data, x='vg_difficulty', y='vb_difficulty',
                 legend_label="Local Courses",
                 line_color='red', fill_color='red', size=6)

  my_plot.legend.location = 'top_left'
  if filename:
    bokeh.plotting.save(my_plot)

  return my_plot


################## Main Program ########################

FLAGS = flags.FLAGS
flags.DEFINE_integer('chains', 36, 'Number of MCMC chains to explore',
                     lower_bound=1)
flags.DEFINE_integer('draws', 2000, 'Number of draws to make when sampling',
                     lower_bound=1)
flags.DEFINE_string('data_dir', DEFAULT_DATA_DIR,
                    'Where to find the raw data (course, runner, date, time).')
flags.DEFINE_string('result_dir', DEFAULT_DATA_DIR,
                    'Where to store the plots we generate.')
flags.DEFINE_string('cache_dir', '',
                    'Where to cache the analysis results.')
flags.DEFINE_bool('use_cached_model', True,
                  'Use the precomputed (cached) model')
flags.DEFINE_integer('seed', -1, 'Initial random seed for entire program.'
                     'A megative value means do not initialze.')
flags.DEFINE_string('monthly_spec', 'normal,0,100',
                    'Model and params for the monthly model')
flags.DEFINE_string('yearly_spec', 'normal,0,100',
                    'Model and params for the yearly model')
flags.DEFINE_string('course_spec', 'normal,1,1',
                    'Model and params for the course model')
flags.DEFINE_string('runner_spec', 'normal,1,1',
                    'Model and params for the runner model')
flags.DEFINE_enum('genders', 'both', ['boys', 'girls', 'both'],
                  'Which genders to train and test.')
flags.DEFINE_string('flag_filename', 'flag_values.txt',
                    'Where to store a record of the flags used for this test')
flags.DEFINE_string('boys_data', 'boys_v2.csv', 
                    'Where to read the race results for the boys.')
flags.DEFINE_string('girls_data', 'girls_v2.csv', 
                    'Where to read the race results for the girls.')


def print_flags(filename: str):
  with open(filename, 'w') as fp:
    flag_dict = flags.FLAGS.flag_values_dict()
    for k, v in flag_dict.items():
      fp.write(f'{k}: {v}\n')


def main(_):
  start_time = time.time()
  print(f'Have {os.cpu_count()} CPUs available for this job.')
  os.makedirs(FLAGS.result_dir, exist_ok=True)
  if FLAGS.flag_filename:
    print_flags(os.path.join(FLAGS.result_dir, FLAGS.flag_filename))

  if FLAGS.seed >= 0:
    # https://discourse.pymc.io/t/how-to-set-a-seed-for-pm-sample/11497
    rng = np.random.default_rng(FLAGS.seed)
  else:
    rng = None
  vb_data = import_xcstats(FLAGS.boys_data)
  if FLAGS.boys_data == FLAGS.girls_data:
    vg_data = vb_data[(vb_data['gender'] == 'g') | (vb_data['gender'] == 'G')]
    vb_data = vb_data[(vb_data['gender'] == 'b') | (vb_data['gender'] == 'B')]
  else:
    vg_data = import_xcstats(FLAGS.girls_data)

  assert len(vb_data[(vb_data['gender'] == 'g') | (vb_data['gender'] == 'G')]) == 0
  assert len(vg_data[(vg_data['gender'] == 'b') | (vg_data['gender'] == 'B')]) == 0
  print(f'Read in {vb_data.shape[0]} boys and '
        f'{vg_data.shape[0]} girls results')

  # Work with the top 25% of results for now.
  top_runner_percent = 25

  if FLAGS.genders in ('both', 'boys'):
    cache_file = os.path.join(FLAGS.cache_dir, 'vb_analysis.pickle')
    if FLAGS.use_cached_model and FLAGS.cache_dir and os.path.exists(cache_file):
      print('\nLoading boys model from cache')
      (vb_xc_model, vb_model_trace,
      top_runner_percent, vb_select,
      vb_course_id_to_index, vb_runner_id_to_index, vb_course_id_to_name,
      vb_map_estimate) = load_model(cache_file, None)
    else:
      vb_course_id_to_name = find_course_course_id_to_names(vb_data)
      vb_select, vb_runner_id_to_index, vb_course_id_to_index = prepare_xc_data(
          vb_data, place_fraction=top_runner_percent/100.0)
      print('\nBuilding boys model...')
      vb_xc_model, vb_map_estimate, vb_model_trace = build_and_test_model(
        vb_select, FLAGS.monthly_spec, FLAGS.yearly_spec,
        FLAGS.course_spec, FLAGS.runner_spec, chains=FLAGS.chains,
        draws=FLAGS.draws, seed=rng)
      if FLAGS.cache_dir:
        os.makedirs(FLAGS.cache_dir, exist_ok=True)
        save_model(cache_file,
                  vb_xc_model, vb_model_trace, vb_map_estimate,
                  top_runner_percent, vb_select,
                  vb_course_id_to_index, vb_runner_id_to_index, vb_course_id_to_name,
                  default_dir=None)
    print(vb_map_estimate)
  else:
    print('Not building boys model.')
  print(f'Boys model has {len(vb_runner_id_to_index)} runners and '
        f'{len(vb_course_id_to_index)} courses.')

  if FLAGS.genders in ('both', 'girls'):
    cache_file = os.path.join(FLAGS.cache_dir, 'vg_analysis.pickle')
    if FLAGS.use_cached_model and FLAGS.cache_dir and os.path.exists(cache_file):
      print('\nLoading girls model from cache')
      (vg_xc_model, vg_model_trace,
      top_runner_percent, vg_select,
      vg_course_id_to_index, vg_runner_id_to_index, vg_course_id_to_name,
      vg_map_estimate) = load_model(cache_file, None)
    else:
      vg_course_id_to_name = find_course_course_id_to_names(vg_data)
      vg_select, vg_runner_id_to_index, vg_course_id_to_index = prepare_xc_data(
          vg_data, place_fraction=top_runner_percent/100.0)
      print('\nBuilding girls model...')
      vg_xc_model, vg_map_estimate, vg_model_trace = build_and_test_model(
        vg_select, FLAGS.monthly_spec, FLAGS.yearly_spec,
        FLAGS.course_spec, FLAGS.runner_spec, chains=FLAGS.chains,
        draws=FLAGS.draws, seed=rng)
      if FLAGS.cache_dir:
        os.makedirs(FLAGS.cache_dir, exist_ok=True)
        save_model(cache_file,
                  vg_xc_model, vg_model_trace, vg_map_estimate,
                  top_runner_percent, vg_select,
                  vg_course_id_to_index, vg_runner_id_to_index, vg_course_id_to_name,
                  default_dir=None)
    print(vg_map_estimate)
  else:
    print('Not building girls model.')
  print(f'Girls model has {len(vg_runner_id_to_index)} runners and '
        f'{len(vg_course_id_to_index)} courses.')

  ##################### Plot all the (VB) results.  ####################
  plot_map_course_difficulties(
      vb_map_estimate,
      title='Histogram of VB Course Difficulties (MAP)',
      filename=os.path.join(FLAGS.result_dir,
                            'vb_map_course_difficulties.png'))

  vb_monthly_trace_means = plot_monthly_slope_predictions(
      vb_model_trace,
      title='Histogram of VB Monthly Slope Predictions',
      filename=os.path.join(FLAGS.result_dir, 'vb_monthly_slope.png'))

  vb_yearly_trace_means = plot_yearly_slope_predictions(
      vb_model_trace,
      title='Histogram of VB Yearly Slope Predictions',
      filename=os.path.join(FLAGS.result_dir, 'vb_yearly_slope.png'))

  vb_difficulty_trace_slopes = plot_map_trace_difficulty_comparison(
      vb_map_estimate,
      vb_model_trace,
      title='Comparison of VB Course Difficulty Estimates',
      filename=os.path.join(FLAGS.result_dir,
                            'vb_course_difficulty_comparison.png'))

  if vb_monthly_trace_means and vb_yearly_trace_means:
    plot_year_month_difficulty_tradeoff(
        vb_monthly_trace_means,
        vb_yearly_trace_means,
        vb_difficulty_trace_slopes,
        title='Tradeoff between VB year/month and course difficulties',
        filename=os.path.join(FLAGS.result_dir,
                              'vb_year_month_course_tradeoff.png'))

    # Plot number of races per (VB) runner.
    print(vb_select.head())
    boy_counts = vb_select.groupby(['runnerID']).size()
    plt.clf()
    plt.hist(boy_counts)
    plt.xlabel('Number of Races')
    plt.title('VB Race Counts per Runner')
    filename = os.path.join(FLAGS.result_dir, 'vb_race_frequency_histogram.png')
    plt.savefig(filename)

  ##################### Plot all the (VG) results.  ####################
  plot_map_course_difficulties(
      vg_map_estimate,
      title='Histogram of VG Course Difficulties (MAP)',
      filename=os.path.join(FLAGS.result_dir,
                            'vg_map_course_difficulties.png'))

  vg_monthly_trace_means = plot_monthly_slope_predictions(
      vg_model_trace,
      title='Histogram of VG Monthly Slope Predictions',
      filename=os.path.join(FLAGS.result_dir, 'vg_monthly_slope.png'))

  vg_yearly_trace_means = plot_yearly_slope_predictions(
      vg_model_trace,
      title='Histogram of VG Yearly Slope Predictions',
      filename=os.path.join(FLAGS.result_dir, 'vg_yearly_slope.png'))

  vg_difficulty_trace_slopes = plot_map_trace_difficulty_comparison(
      vg_map_estimate,
      vg_model_trace,
      title='Comparison of VG Course Difficulty Estimates',
      filename=os.path.join(FLAGS.result_dir,
                            'vg_course_difficulty_comparison.png'))

  if vg_monthly_trace_means and vg_yearly_trace_means:
    plot_year_month_difficulty_tradeoff(
        vg_monthly_trace_means,
        vg_yearly_trace_means,
        vg_difficulty_trace_slopes,
        title='Tradeoff between VG year/month and course difficulties',
        filename=os.path.join(FLAGS.result_dir,
                              'vg_year_month_course_tradeoff.png'))

    # Plot number of races per (VG) runner.
    girl_counts = vb_select.groupby(['runnerID']).size()
    plt.clf()
    plt.hist(girl_counts)
    plt.xlabel('Number of Races')
    plt.title('VG Race Counts per Runner')
    filename = os.path.join(FLAGS.result_dir, 'vg_race_frequency_histogram.png')
    plt.savefig(filename)

  ################## Create course difficulty summary tables ###################
  local_course_list = find_local_courses(vb_data)
  scatter_df = create_result_frame(vb_data, vg_data,
                                  vb_course_id_to_index, vg_course_id_to_index,
                                  vb_course_id_to_name,
                                  vb_model_trace, vg_model_trace,
                                  local_course_list)

  local_df = scatter_df[scatter_df['local_course']].sort_values('vb_difficulty')
  table_title = f'Bay Area Course Difficulties ({local_df.shape[0]} courses)'
  create_html_table(
    local_df,
    os.path.join(FLAGS.result_dir, 'course_difficulties_local.html'),
    title=table_title)

  create_markdown_table(local_df)

  table_title = ('California Course Difficulties '
                f'({scatter_df.shape[0]} courses)')
  create_html_table(
      scatter_df.sort_values('vb_difficulty'),
      os.path.join(FLAGS.result_dir, 'course_difficulties.html'),
      title=table_title)

  filename = os.path.join(FLAGS.result_dir,
                          'vb_vg_difficulties_comparison.html')
  plot_difficulty_comparison(scatter_df, filename)

  create_hank_correction_list(scatter_df,
                              os.path.join(FLAGS.result_dir,
                                           'hank_corrections.txt'))

  ######################  Check prediction quality #####################
  y_pred = pm.sample_posterior_predictive(vb_model_trace,
                                          model=vb_xc_model)
  observed = y_pred['observed_data']['y_like'].values
  predictions = np.mean(y_pred['posterior_predictive']['y_like'].values,
                        axis=(0, 1))  # Average over chains and draws
  y_error = (observed - predictions)/observed*100
  y_error_std = np.std(y_error)
  job_description = (f'{FLAGS.monthly_spec}/{FLAGS.yearly_spec}/'
                    f'{FLAGS.course_spec}/{FLAGS.runner_spec}')
  print(f'\nStandard deviation of prediction errors is {y_error_std}% for '
        f'{job_description}')
  plt.clf()
  plt.hist(y_error, bins=100)
  plt.xlabel('Prediction Error (%)')
  plt.title('VB Model Prediction Errors')
  filename = os.path.join(FLAGS.result_dir, 'vb_prediction_error_histogram.png')
  plt.savefig(filename)

  print(f'All done after {(time.time()-start_time)/60.0} minutes.')


if __name__ == '__main__':
  app.run(main)
