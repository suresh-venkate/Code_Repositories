cd datasets
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz	
tar -xvzf cars_train.tgz
tar -xvzf cars_test.tgz
tar -xvzf car_devkit.tgz