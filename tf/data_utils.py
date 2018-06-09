import numpy as np
import string
import random
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

def listFind(aList, begin, elem):
	for i in range(begin, len(aList)):
		if aList[i] == elem:
			return i
	return -1

# Read the relations in relation2IdFileName
# and return a dictionary that maps a 
# string(relation) to an int(id)
def readRelations(relation2IdFileName):
	relation2Id = {}
	with open(relation2IdFileName) as f:
		for line in f:
			tokens = line.split()
			relation2Id[tokens[0]] = tokens[1]
	return relation2Id

def addToken(index, tokens, word2Id, id2Word, counter):
	curWord = tokens[index]
	if curWord not in word2Id:
		word2Id[curWord] = counter
		id2Word[counter] = curWord
		counter += 1
	return " " + str(word2Id[curWord]), counter

def getMax(d):
	maxV = None
	for k, v in d.items():
		v = int(v)
		if maxV is None:
			maxV = v
		else:
			if v > maxV:
				maxV = v
	return maxV

def addToRelations(relation2IdFileName, relation, maxIndex, relation2Id):
	maxIndex += 1
	relation2Id[relation] = maxIndex
	with open(relation2IdFileName, 'a') as f:
		s = relation + " " + str(maxIndex) + "\n"
		f.write(s)
	return maxIndex

# Convert the fileName in the format(per line):
# [e1_id e2_id e1 e2 relation word1 word2 ...]
# [relation_id word1_id word2_id ... e1 ... e2 ...]
def convertStringsToNumbers(fileName, newFileName, relation2IdFileName):
	relation2Id = readRelations(relation2IdFileName)
	maxIndex = getMax(relation2Id)
	newFile = open(newFileName, "w+")
	word2Id = {}
	id2Word = {}
	counter = 0
	c = 0
	with open(fileName) as f:
		for line in f:
			lineToWrite = ""
			tokens = line.split()
			if tokens[4] not in relation2Id:
				maxIndex = addToRelations(relation2IdFileName, tokens[4], maxIndex, relation2Id)
			lineToWrite += str(relation2Id[tokens[4]])
			e1 = tokens[2]
			e2 = tokens[3]
			#print(e1, e2)
			i1 = listFind(tokens, 5, e2)
			i2 = listFind(tokens, i1, e1)
			if i1 == -1 :
				#c += 1
				continue
			# If the two entities don't occur together
			if i2 == -1:
				i2 = listFind(tokens, 5, e1)
				if i2 == -1:
					c += 1
					continue
			for i in range(5, i1):
				add, counter = addToken(i, tokens, word2Id, id2Word, counter)
				lineToWrite += add
			lineToWrite += " " + e2
			for i in range(i1+1, i2):
				add, counter = addToken(i, tokens, word2Id, id2Word, counter)
				lineToWrite += add
			lineToWrite += " " + e1
			for i in range(i2+1, len(tokens) - 1):
				add, counter = addToken(i, tokens, word2Id, id2Word, counter)
				lineToWrite += add
		print(c)
			
#convertStringsToNumbers("../data/train.txt", "../data/newTrain.txt", "../data/relation2id.txt")

def test(fileName="test.txt"):
	for i in range(0, 8):
		with open(fileName, 'a') as f:
			f.write(str(i))
#test()


# Per line structure:
# [e1_id e2_id e1 e2 relation word1 word2 ...]
# if the second entity does not always appear before the first one, raise an error
def analysis(fileName):
	counter = 0
	secondBeforeFirst = 0
	firstBeforeSecond = 0
	secondBeforeFirstAnd = 0
	with open(fileName) as f:
		for line in f:
			tokens = line.split()
			e1 = tokens[2]
			e2 = tokens[3]
			relation = tokens[4]
			i1 = listFind(tokens, 5, e1)
			i2 = listFind(tokens, 5, e2)
			if i1 == -1 or i2 == -1:
				counter += 1
				continue
			i1 = listFind(tokens, 5, e1)
			i2 = listFind(tokens, i1 + 1, e2)
			if i2 != -1:
				firstBeforeSecond += 1
				i1 = listFind(tokens, 5, e2)
				i2 = listFind(tokens, i1 + 1, e1)
				if i2 != -1:
					secondBeforeFirstAnd += 1
			i1 = listFind(tokens, 5, e2)
			i2 = listFind(tokens, i1 + 1, e1)
			if i2 != -1:
				secondBeforeFirst += 1
	print("There are %d instances where at least one of the entities do not occur in the sentence", counter)
	print("There are %d instances where the second entity exists after the first entity", firstBeforeSecond)
	print("There are %d instances where the second entity exists before the first entity", secondBeforeFirst)
	print("There are %d instances where the first entity exists before the second entity and second before the first", secondBeforeFirstAnd)

def analysis2(fileName):
	counter = 0
	with open(fileName) as f:
		for line in f:
			tokens = line.split()
			e1 = tokens[2]
			e2 = tokens[3]
			relation = tokens[4]
			idx = listFind(tokens, 5, e1)
			if idx != -1:
				idx = listFind(tokens, idx + 1, e1)
				if idx != -1:
					print(line)
					counter += 1 
					if counter == 5:
						return

def analysis3(fileName, gloveFileName="../../../../../../GitHub/cs224u/vsmdata/glove/glove.6B.50d.txt"):
	included = 0
	count = []
	vocab = utils.glove2dict(gloveFileName)  # dict[word] -> numpy array(embed_dim,)
	with open(fileName) as f:
		lineNum = 0
		for line in f:
			if lineNum % 4 != 0:
				lineNum += 1
				continue
			firstStartIdx = line.find('<e1>') + 4
			firstEndIdx = line.find('</e1>')
			secondStartIdx = line.find('<e2>') + 4
			secondEndIdx = line.find('</e2>')
			if firstStartIdx == -1 or firstEndIdx == -1 or secondStartIdx == -1 or secondEndIdx == -1:
				print("ERROR")
			firstEntity = line[firstStartIdx:firstEndIdx]
			secondEntity = line[secondStartIdx:secondEndIdx]
			#print(firstEntity, secondEntity)
			if firstEntity not in vocab:
				count.append(firstEntity)
			else:
				included += 1
			if secondEntity not in vocab:
				count.append(secondEntity)
			else:
				included += 1
			lineNum += 1
	print(included)
	print(len(count))
	#print(count)

#analysis3("SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT")


def createWord2Index(vocab, word2Index):
	counter = 0
	for word in vocab:
		word2Index[word] = counter
		counter += 1

# Adds "<UNK>" token to both vocab and word2Index
def addUnk(vocab, word2Index):
	word2Index["<UNK>"] = len(word2Index)
	unkValue = []
	# pick unkValue as random values
	for i in range(0, vocab["the"].size):
		randomKey = random.choice(list(vocab.keys()))
		unkValue.append(vocab[randomKey][i])  
	vocab["<UNK>"] = np.asarray(unkValue)

# Adds "<PAD>" token to both vocab and word2Index
def addPadding(vocab, word2Index):
	word2Index["<PAD>"] = len(word2Index)
	vocab["<PAD>"] = np.zeros_like(vocab["the"])

def getLineCount(line):
	tokens = line.split()
	count = 0
	within = False
	for token in tokens:
		if "<e1>" in token or "<e2>" in token:
			within = True
		if "</e1>" in token or "</e2>" in token:
			within = False
		if not within:
			count += 1
	return count

#print(getLineCount("8000	\"The <e1>surgeon</e1> cuts a small <e2>hole</e2> in the skull and lifts the edge of the brain to expose the nerve.\""))

def reverseDictFile(newFileName, fileName="SemEval2010_task8_all_data/cleaned/cleaned_word2Index.txt"):
	newFile = open(newFileName, 'w+')
	with open(fileName) as f:
		for line in f:
			tokens = line.split()
			newFile.write(tokens[1] + " " + tokens[0] + "\n")


def writeDictToFile(d):
	file = open("SemEval2010_task8_all_data/cleaned/cleaned_word2Index.txt", 'w+')
	for key in d:
		file.write(key + " " + str(d[key]) + "\n")
	file.close()

def readDictFromFile(d, fileName):
	with open(fileName) as f:
		for line in f:
			tokens = line.split()
			d[tokens[0]] = tokens[1]

'''
Intended for the new dataset(SemEval2010_task8)
Given a training instance in the form of:
id 	"word1 word2 ... <e1>entity1_1 entity1_2 ... entity1_n</e1> ... <e2>entity2_1 ... entity2_m</e2> ... wordk."
relation(ei, ej)
Comment:
EMPTY LINE
Convert it to:
id relation_class_no word1_index word2_index ... <e1> ... <e2> ... wordk_index
and write these to a new file called cleaned_train.txt. Also, write the word2Index dict to cleaned_word2Index.txt
'''
def convertSentencesToIdx(relation2IdFileName="SemEval2010_task8_all_data/cleaned/cleaned_entity2Id.txt", trainFileName="SemEval2010_task8_all_data/SemEval2010_task8_training/small_TRAIN_FILE.TXT", gloveFileName="../../../../../../GitHub/cs224u/vsmdata/glove/glove.6B.50d.txt"):
	vocab = utils.glove2dict(gloveFileName)  # dict[word] -> numpy array(embed_dim,)
	word2Index = {}
	createWord2Index(vocab, word2Index)
	addUnk(vocab, word2Index)
	addPadding(vocab, word2Index)
	writeDictToFile(word2Index)
	relation2Id = {}
	readDictFromFile(relation2Id, relation2IdFileName)
	count = 0
	cleanedFile = open("SemEval2010_task8_all_data/cleaned/cleaned_train.txt", 'w+')
	with open(trainFileName) as f:
		lineNum = 0
		table = str.maketrans({key: None for key in string.punctuation if key != "-"})
		for line in f:
			if lineNum % 4 == 2 or lineNum % 4 == 3:
				lineNum += 1
				continue
			elif lineNum % 4 == 1:
				lineNum += 1
				lineToWrite = relation2Id[line[:line.find("(")]] + lineToWrite
				cleanedFile.write(lineToWrite)
				continue
			lineCount = getLineCount(line)
			# Remove all the words in the middle of brackets
			firstStartIdx = line.find('<e1>') + len("<e1>")
			firstEndIdx = line.find('</e1>')
			firstEntity = line[firstStartIdx:firstEndIdx + 5] 
			line = line.replace(firstEntity, "") 
			secondStartIdx = line.find('<e2>') + len("<e2>")
			secondEndIdx = line.find('</e2>')
			secondEntity = line[secondStartIdx:secondEndIdx + 5]
			line = line.replace(secondEntity, "")
			line = line.translate(table)
			line = line.lower()
			tokens = line.split()
			lineToWrite = ""
			# Write sentence id
			lineToWrite += " " + str(tokens[0])
			for token in tokens[1:]:
				lineToWrite += " "
				if "e1" in token:
					lineToWrite += "<e1>"
				elif "e2" in token:
					lineToWrite += "<e2>"
				elif token in word2Index:
					lineToWrite += str(word2Index[token])
				else:
					lineToWrite += str(word2Index["<UNK>"])
			lineToWrite += ("\n")
			lineNum += 1
			'''
			# Sanity check:
			if "<e1>" not in lineToWrite or "<e2>" not in lineToWrite:
				print("ERROR")
				print(lineToWrite)
			if lineCount != len(lineToWrite.split()):
				count += 1
				print(lineCount)
				print(len(lineToWrite.split()))
				print("ERROR")
				print(lineToWrite)
			'''
	cleanedFile.close()
	return vocab, word2Index, relation2Id


#convertSentencesToIdx() 
#reverseDictFile("SemEval2010_task8_all_data/cleaned/cleaned_index2Word.txt")

#analysis("../data/train.txt")




