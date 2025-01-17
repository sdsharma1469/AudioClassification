# from audioset_download import Downloader
# d = Downloader(root_path='test/Siren', labels=["Siren",], n_jobs=5, download_type='unbalanced_train', copy_and_replicate=False)

# d.download(format = 'vorbis')
import soundata 

dataset = soundata.initialize('urbansound8k')
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data