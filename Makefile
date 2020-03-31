all:
	python frac_diff_sp500.py
	python frac_diff_x.py
	convert -delay 10 -loop 0 img/*png anim.gif
