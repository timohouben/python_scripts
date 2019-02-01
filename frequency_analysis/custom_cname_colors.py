#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:51:58 2019

@author: houben


"""

import random
import matplotlib


colors = matplotlib.colors.cnames













def init_colors(self):
        'initialize all_colors'

        self.all_colors = []

        # remove any colors with 'white' or 'yellow in the name
        skip_colors_substrings = ['white', 'yellow']
        skip_colors_exact = ['black', 'red', 'blue']

        for col in matplotlib.colors.cnames:
            skip = False

            for col_substring in skip_colors_substrings:
                if col_substring in col:
                    skip = True
                    break

            if not skip and not col in skip_colors_exact:
                self.all_colors.append(col)

        # we'll re-add these later; remove them before shuffling
        first_colors = ['lime', 'cyan', 'orange', 'magenta', 'green']

        for col in first_colors:
            self.all_colors.remove(col)

        # deterministic shuffle of all remaining colors
        random.seed(0)
        random.shuffle(self.all_colors)

        # prepend first_colors so they get used first
        self.all_colors = first_colors + self.all_colors 