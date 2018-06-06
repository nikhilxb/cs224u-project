import os
import random

testFileName = "./testfile.txt"
devFileName = "./devFile.txt"
newTestFileName = "./newTestFile.txt"

def splitTestFileIntoTestAndVal(testFileName=testFileName):
	devFile = open(devFileName, "w+")
	tempTestFile = open(newTestFileName, "w+")
	with open(testFileName) as testFile:
		for line in testFile:
			# add to testFile
			if random.uniform(0, 1) <= 0.5:
				tempTestFile.write(line)
			# add to dev file
			else:
				devFile.write(line)
	# close both file
	tempTestFile.close()
	devFile.close()


def main():
	splitTestFileIntoTestAndVal()


if __name__ == "__main__":
	main()
