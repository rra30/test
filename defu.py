import os
import streamlit as st

def list_files(startpath):
    tree_structure = ""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree_structure += f"{indent}{os.path.basename(root)}/\n"
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree_structure += f"{subindent}{f}\n"
    return tree_structure
