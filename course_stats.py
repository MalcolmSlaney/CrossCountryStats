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