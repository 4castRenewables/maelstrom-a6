import a6.plotting.animation as animation


def test_animate_timeseries(da):
    animation.animate_timeseries(da, display=False)
