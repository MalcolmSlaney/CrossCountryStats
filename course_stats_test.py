from absl.testing import absltest
import numpy as np
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

    plt.plot(df.race_month.values, df.times.values, 'x')
    plt.xlabel('Months into season');
    plt.ylabel('Simulated course time (s)');

    # Each line is a different course.
    plt.savefig('Results/simulated_data_month_year.png')
    print('Finished savefig')
      
    # Build the model
    xc_model = course_stats.create_xc_model(df, use_course=False, 
                                            use_runner=False)
    
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
                                                      use_year=False)
    xc_model = course_stats.create_xc_model(df, use_runner=False, 
                                            use_year=False)

    model_trace = pm.sample(model=xc_model)
    map_estimate = pm.find_MAP(model=xc_model)

    plt.clf()
    plt.plot(course_difficulties, map_estimate['course_est'], 'x');
    plt.savefig('Results/simulated_data_month_course_trace.png')
    print(f'MAP Course difficulties: {map_estimate["course_est"]}')

    d = np.mean(model_trace.posterior.course_est[1].values, axis=0)
    print(f'Trace Course difficulties: {d}')

    # Make sure MAP and trace mean agree.
    np.testing.assert_array_almost_equal(map_estimate['course_est'], d,
                                         decimal=1)
    # Make sure MAP estimate matches the expected (synthetic) difficulties
    np.testing.assert_array_almost_equal(map_estimate['course_est'], 
                                         course_difficulties, decimal=1)
                                         
  def test_everything(self):
    (df,
     course_difficulties,
     runner_abilities) = course_stats.generate_xc_data(noise=10)
    xc_model = course_stats.create_xc_model(df)
    map_estimate = pm.find_MAP(model=xc_model)

    plt.clf()
    plt.plot(runner_abilities, map_estimate['runner_est'], 'x')
    plt.xlabel('Simulated Ground Truth')
    plt.ylabel('MAP Estimate')
    plt.savefig('Results/simulated_data_everything_runner_abilities.png')

if __name__ == '__main__':
  absltest.main()