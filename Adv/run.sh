pip install tensorflow[and-cuda]
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip install seaborn scikit-learn kaggle matplotlib numpy
pip install transformers==4.44.2 torch tf-keras
kaggle datasets download -d shuvoalok/raf-db-dataset
kaggle datasets download -d msambare/fer2013
unzip raf-db-dataset.zip -d RAF-DB
unzip fer2013.zip -d FER2013
rm raf-db-dataset.zip fer2013.zip