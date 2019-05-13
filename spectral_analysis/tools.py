def get_ogs_folders(path):
    """
    Returns a list of directories where OGS model runs have been setup. It does not decide whether the model has run or not.

    Parameters
    ----------
    path : string
        Path to multiple OGS project folders.

    Yields
    ------
    project_folder_list : list of strings
        Containing all folder names where OGS runs have been set up.
    """

    import os
    import glob

    file_extensions_list = ["*.gli", "*.msh", "*.out", "*.pcs", "*.num"]
    project_folder_list = [
        f for f in os.listdir(str(path)) if not f.startswith(".")
    ]

    print(project_folder_list)
    for folder in project_folder_list:
        print(folder)
        check_extensions = []
        print(len(project_folder_list))
        for extension in file_extensions_list:
            if glob.glob(path + "/" + folder + "/" + extension):
                check_extensions.append(True)

        if len(check_extensions) != len(file_extensions_list):
            project_folder_list.remove(folder)

get_ogs_folders("/Users/houben/Desktop")

THIS FUNCTION ISNT WORKING FINE!!!! I DONT KNOW YET Y
