import urllib
import os

testfile = urllib.URLopener()
testfile.retrieve("https://raw.githubusercontent.com/Kristian60/NetworkRisk/master/Connectedness.py", "newGit.py")

os.system('python newGit.py')