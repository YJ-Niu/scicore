##########################################################################
# This file serves as the dataset registry for SciPy Datasets SubModule.
##########################################################################


# To generate the SHA256 hash, use the command
# openssl sha256 <filename>
registry = {
    "ascent.dat": "03ce124c1afc880f87b55f6b061110e2e1e939679184f5614e38dacc6c1957e2",
    "ecg.dat": "f20ad3365fb9b7f845d0e5c48b6fe67081377ee466c3a220b7f69f35c8958baf",
    "face.dat": "9d8b0b4d081313e2b485748c770472e5a95ed1738146883d84c7030493e82886"
}

# The GitHub contents API is used instead of raw.githubusercontent.com:
# with the Accept: application/vnd.github.raw+json header (sent by
# fetch_data) it returns the raw file bytes directly, and it is reachable
# from network environments where the raw domain is not.
registry_urls = {
    "ascent.dat": "https://api.github.com/repos/scipy/dataset-ascent/contents/ascent.dat",
    "ecg.dat": "https://api.github.com/repos/scipy/dataset-ecg/contents/ecg.dat",
    "face.dat": "https://api.github.com/repos/scipy/dataset-face/contents/face.dat"
}

# dataset method mapping with their associated filenames
# <method_name> : ["filename1", "filename2", ...]
method_files_map = {
    "ascent": ["ascent.dat"],
    "electrocardiogram": ["ecg.dat"],
    "face": ["face.dat"]
}
