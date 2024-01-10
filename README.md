This code models the difficulties of different cross country (XC) courses. 
The goal is to provide a single number that models each course's difficulty 
and can be used to adjust expected race times across courses.

Using XC results, it builds a model that takes into account
these different factors:

  * Runner's inate ability
  * Course difficulty
  * Runner's month-to-month improvement over the season
  * Runner's year-to-year improvement over their career

The model estimates a single number (ability or difficulty) for 
each runner and course.  While the month-to-month and year-to-year parameters
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
