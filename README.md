## Introduction
This code models the difficulties of different cross country (XC) courses. 
The goal is to provide a single number that models each course's difficulty 
and can be used to adjust expected race times across courses.
For example, course A might be 1.1 times harder than course B,
and thus the expected times for course A will be 10% higher.

Using XC race results over multiple years and courses,
it builds a model that takes into account these different factors:

  * Course difficulty
  * Runner's inate ability
  * Runner's month-to-month improvement over the season
  * Runner's year-to-year improvement over their career

The model estimates a single number (ability or difficulty) for 
each runner and each course.
While the month-to-month and year-to-year parameters
are averages that apply to all runners.

To be more specific, given each runner's race times, the model fits parameters
to a model that looks like this:

race_time = (average_race_time - race_month\*month_slope - student_year\*year_slope)\*runner_ability*course_difficulty

Here, the race times are in seconds.
The race_month is the numerical month, starting with September is 0.
The student_year is the high-school year of the student, where freshman is 0.
Thus the slopes are in terms of seconds per month or year, for easier interprability.

The runner_ability and course_difficulty parameters are multiplicative factors
that modify the expected race time during the season. Both factors adjust the
expected times, but in different fashions.  So higher (>1) course difficulties
represent *harder* courses.  While lower (<1) runner abilities represent *faster*
runners. In both cases, higher numbers represent longer finish times.

Note, the outputs from this model are unnormalizedm and should be considered
relative results. 
While both the ability and difficulty numbers tend to be close to 1, 
the baselines are arbitrary.
Thus an average course_difficulty of 0.5 and an average runner ability of 2 will
produce the same overall race-time predictions as the reverse.

## Bayesian Modeling
Given all the data, we use a Bayesian framework to fit the model parameteres 
to the data.