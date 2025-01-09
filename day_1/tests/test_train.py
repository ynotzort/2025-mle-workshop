from datetime import datetime
import os

from duration_prediction.train import train


class TestTraining:
    def test_training_regression_value(self):
        train_date = datetime(2022, 1, 1)
        validation_date = datetime(2022, 2, 1)
        
        output_file = "/tmp/__test_training_regression_value.bin"
        mse = train(train_date, validation_date, output_file)
        
        # test if regression value is the same as last time
        assert abs(mse - 8.1893) < 0.001
        
    def test_training_creates_a_file(self):
        train_date = datetime(2022, 1, 1)
        validation_date = datetime(2022, 2, 1)
        
        output_file = "/tmp/__test_training_creates_a_file.bin"
        try:
            os.remove(output_file)
        except FileNotFoundError:
            # file was not existing but this is fine
            ...
        assert not os.path.exists(output_file)

        _ = train(train_date, validation_date, output_file)

        # test if file is created
        assert os.path.exists(output_file)
        
        # clean up
        os.remove(output_file)