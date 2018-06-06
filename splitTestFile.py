import random

def splitTestFileIntoTestAndVal(testFileName):
    devFile = open("data/val.txt", "w+")
    tempTestFile = open("data/new_test.txt", "w+")
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

if __name__ == "__main__":
    splitTestFileIntoTestAndVal('data/test.txt')
