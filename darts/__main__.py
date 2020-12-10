import warnings
from darts import app

if __name__ == '__main__':
    # catch warnings from np.polyfit for poor line intersections
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app.run()
    