NBA-hair
========

Hair identification in images of NBA players

The idea behind this is to use python to identify (and quantify?) hair in images.  The data set was selected because of its simplicity - all faces are centered and facing forward and there is no background.

Plan - 
  1. parse list of NBA players
  2. get images from web
  3. identify and trace hair (or identify that there is no hair)
  4. use ML for quality control
  5. look for correlation with other player stats

Current status - 
  Hair identification is pretty solid using contrast filtering
  
Python reqs:
  scipy, numpy, pandas, matplotlib, PIL
