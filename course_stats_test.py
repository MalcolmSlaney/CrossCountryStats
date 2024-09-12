from absl.testing import absltest
import numpy as np
import os
import pymc as pm
import matplotlib.pyplot as plt

import course_stats


class XCStatsTest(absltest.TestCase):
  """Test cases for QuickSIN Google Cloud ASR tests."""

  def test_month_year_model(self):
    # Generate data with course and months only.
    df, _, _ = course_stats.generate_xc_data(noise=1, use_course=False, 
                                             standard_time=1080,
                                             monthly_improvement=10,
                                             yearly_improvement=20,
                                             use_runner=False, num_runners=20)

    plt.clf()
    plt.plot(df.race_month.values, df.times.values, 'x')
    plt.xlabel('Months into season');
    plt.ylabel('Simulated course time (s)');
    plt.title('Simulated Race Times over Four HS Years')
    # Each line is a different course.
    plt.savefig('Results/simulated_data_month_year.png')
    print('Finished savefig')
      
    # Build the model
    xc_model = course_stats.create_xc_model(df, course_spec=None, 
                                            runner_spec=None)
    
    # Fit the model
    trace = pm.sample(model=xc_model)
    plt.clf()
    pm.plot_trace(trace);
    plt.savefig('Results/simulated_data_month_year_trace.png')

    # Test the MAP parameters
    map_estimate = pm.find_MAP(model=xc_model)
    print(f'Bias: {map_estimate["bias"]}, eps: {map_estimate["eps"]}')
    print(f'Monthly slope: {map_estimate["monthly_slope"]}, '
          f'Yearly slope: {map_estimate["yearly_slope"]}')
    
    self.assertAlmostEqual(map_estimate['bias'], 1080, delta=1)
    self.assertAlmostEqual(map_estimate['monthly_slope'], 10, delta=10/100.0)
    self.assertAlmostEqual(map_estimate['yearly_slope'], 20, delta=20/100.0)
    self.assertAlmostEqual(map_estimate['eps'], 0, delta=.2)


  def test_month_course_model(self):
    (df,
    course_difficulties,
    runner_abilities) = course_stats.generate_xc_data(noise=1, use_course=True, 
                                                      use_runner=False, 
                                                      use_month=False,
                                                      use_year=False)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(course_difficulties, '.')
    plt.ylabel('Course Difficulty Factor');
    plt.subplot(1, 2, 2)
    plt.hist(runner_abilities)
    plt.xlabel('Runner Time Factor')
    plt.savefig('Results/simulated_data_month_course_factors.png')

    xc_model = course_stats.create_xc_model(df, 
                                            runner_spec=None, 
                                            month_spec=None, year_spec=None,
                                            course_spec='normal,1,0.1')

    try:
      plt.clf()
      pm.model_to_graphviz(xc_model).render('Results/simulated_data_month_course_graphviz')
    except:
      print('Cannot run graphviz to visualize the Bayesian model.')
      
    model_trace = pm.sample(model=xc_model)
    plt.clf()
    pm.plot_trace(model_trace);
    plt.savefig('Results/simulated_data_month_course_trace.png')
  
    map_estimate = pm.find_MAP(model=xc_model)
    plt.clf()
    plt.plot(course_difficulties, map_estimate['course_est'], 'x');
    plt.xlabel('True Course Difficulty')
    plt.ylabel('MAP Course Difficulty')
    plt.title('Simulated Course Difficulty Result')
    plt.savefig('Results/simulated_data_month_course_map.png')
    print('test_month_course_model: MAP Course difficulties: '
          f'{map_estimate["course_est"]}')

    d = np.mean(model_trace.posterior.course_est[1].values, axis=0)
    print(f'test_month_course_model: Trace Course difficulties: {d}')

    # Make sure MAP and trace mean agree.
    np.testing.assert_array_almost_equal(map_estimate['course_est'], d,
                                         decimal=1)
    # Make sure MAP estimate matches the expected (synthetic) difficulties
    np.testing.assert_array_almost_equal(map_estimate['course_est'], 
                                         course_difficulties, decimal=1)
                                         
  def test_everything(self):
    num_runners = 100
    num_courses = 5
    num_samples = 8000
    (df,
     course_difficulties,
     runner_abilities) = course_stats.generate_xc_data(noise=10, 
                                                       n_samples=num_samples,
                                                       num_courses=num_courses,
                                                       num_runners=num_runners)
    self.assertLen(course_difficulties, num_courses)
    self.assertLen(runner_abilities, num_runners)
    self.assertLen(df.runner_ids.values, num_samples)
    xc_model = course_stats.create_xc_model(df)
    map_estimate = pm.find_MAP(model=xc_model)

    plt.clf()
    plt.plot(runner_abilities, map_estimate['runner_est'], 'x')
    plt.xlabel('Simulated Ground Truth')
    plt.ylabel('MAP Estimate')
    plt.title('Runner Ability Synthetic Test Result')
    plt.savefig('Results/simulated_data_everything_runner_abilities.png')

  def test_all_plots(self):
    default_data_dir = '/tmp'

    (df,
     course_difficulties,
     runner_abilities) = course_stats.generate_xc_data(noise=10)

    xc_model, map_estimate, model_trace = course_stats.build_and_test_model(
        df, monthly_spec='Normal,0,1', yearly_spec='Normal,0,1', 
        course_spec='Normal,0,1', runner_spec='Normal,0,1',
        chains=1, draws=100, tune=100)  # No tuning, just want to see plots.

    monthly_trace_means = course_stats.plot_monthly_slope_predictions(
        model_trace,
        title='Histogram of VB Monthly Slope Predictions',
        filename=os.path.join(default_data_dir, 'monthly_slope.png'))
    self.assertTrue(os.path.exists('/tmp/monthly_slope.png'))
  
    yearly_trace_means = course_stats.plot_yearly_slope_predictions(
        model_trace,
        title='Histogram of VB Yearly Slope Predictions',
        filename=os.path.join(default_data_dir, 'yearly_slope.png'))
    self.assertTrue(os.path.exists('/tmp/yearly_slope.png'))
    
    difficulty_trace_slopes = course_stats.plot_map_trace_difficulty_comparison(
        map_estimate,
        model_trace,
        title='Comparison of VB Course Difficulty Estimates',
        filename=os.path.join(default_data_dir, 
                              'course_difficulty_comparison.png'))
    self.assertTrue(os.path.exists('/tmp/course_difficulty_comparison.png'))
    
    course_stats.plot_year_month_difficulty_tradeoff(
        monthly_trace_means,
        yearly_trace_means,
        difficulty_trace_slopes,
        title='Tradeoff between VB year/month and course difficulties',
        filename=os.path.join(default_data_dir, 
                              'year_month_course_tradeoff.png'))
    self.assertTrue(os.path.exists('/tmp/year_month_course_tradeoff.png'))


if __name__ == '__main__':
  absltest.main()
