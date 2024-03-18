import tensorflow as tf

import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

# Information needed to download files
project_name = 'New-Thesis/Basic'
run_id = 'BAS-2'

# Load specific run that has the model and set mode to read-only
run = neptune.init_run(project=project_name,
                       with_id=run_id, 
                       mode="read-only")

# Download keras file from '/checkpoint/my_model' to current working directory
run['/checkpoint/my_model'].download()

# Download the file to a specific directory
# run["data/sample"].download(destination="path/to/destination")

run.stop()