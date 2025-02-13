from datetime import datetime

# I like this format for adding dates to the end of filenames
getdate = lambda : str(datetime.datetime.now()).replace(' ', '_').replace(':', '.')

