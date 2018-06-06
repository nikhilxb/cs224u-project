import random
import collections


def count(fileName, relations):
    counts = collections.defaultdict(int)
    with open(fileName) as f:
        for line in f:
            relation = line.split()[4]
            if relation in relations:
                counts[relation] += 1
    print(counts)

def sampleExamples(fileName, newFileName):
    # count before
    print("Counts before:")
    count(fileName, {"NA", "/location/location/contains"})
    newFile = open(newFileName, "w+")
    with open(fileName) as f:
        for line in f:
            relation = line.split()[4]
            if relation == "NA":
                if random.uniform(0, 1) <= 0.1:
                    newFile.write(line)
            elif relation == "/location/location/contains":
                if random.uniform(0, 1) <= 0.01:
                    newFile.write(line)
            else:
                newFile.write(line)
    print("Counts after:")
    count(newFileName, {"NA", "/location/location/contains"})



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
    #splitTestFileIntoTestAndVal('data/test.txt')
    sampleExamples('data/train.txt', 'data/newTrain.txt')
