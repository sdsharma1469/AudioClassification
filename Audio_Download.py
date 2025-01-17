from audioset_download import Downloader
d = Downloader(root_path='test', labels=["Siren",], n_jobs=5, download_type='unbalanced_train', copy_and_replicate=False)

d.download(format = 'vorbis')