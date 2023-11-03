import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir_with_slash = os.path.join(parent_dir, '').replace('\\', '/')
rootPath=parent_dir_with_slash+"data/"