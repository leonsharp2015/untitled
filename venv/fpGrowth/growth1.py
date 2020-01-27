
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode  # needs to be updated
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

#----------------------------

def createTree(dataSet, minSup=1):  # create FP-tree from dataset but don't mine
    headerTable = {}
    # go over dataSet twice
    for trans in dataSet:  # first pass counts frequency of occurance
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):  # remove items not meeting minSup
        if headerTable[k] < minSup:#headerTable保存每个单词出现的次数
            del (headerTable[k])
    freqItemSet = set(headerTable.keys())#freqItemSet是出现次数大于阈值的单词集合
    # print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None  # if no items meet min support -->get out
    for k in headerTable:
        #headerTable是dictinary类型：key=单词，value＝[次数,None]
        headerTable[k] = [headerTable[k], None]  # 构造 reformat headerTable to use Node link
    # print 'headerTable: ',headerTable

    retTree = treeNode('Null Set', 1, None)  # create tree
    for tranSet, count in dataSet.items():  # 所有子集合。go through dataset 2nd time
        localD = {} #保存一个子集合内的次数多的单词
        for item in tranSet:  # tranSet子集合内的所有单词   put transaction items in order
            if item in freqItemSet: #在整个集合的所有单词中，出现次数多的单词
                localD[item] = headerTable[item][0] #key=单词，value=个数
        if len(localD) > 0:
            # 子集合内的高频单词按次数排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)  # populate tree with ordered freq itemset
    return retTree, headerTable  # return tree and header table


#headerTable是字典：key=单词，value＝[次数,None(同名的一个节点)]
def updateTree(items, inTree, headerTable, count):#将子集合内的高频单词，形成树:一个子集合内的高频单词，形成父节点－》子节点－》子节点
    key=items[0] #高频单词
    if key in inTree.children:  # check if orderedItems[0] in retTree.children
        inTree.children[key].inc(count)  # incrament count
    else:  # add items[0] to inTree.children
        inTree.children[key] = treeNode(key, count, inTree) #生成items［0］子节点
        if headerTable[key][1] == None:  # update header table
            headerTable[key][1] = inTree.children[key] #更新headerTable的value头指针
        else:
            updateHeader(headerTable[key][1], inTree.children[key])
    if len(items) > 1:  # call updateTree() with remaining ordered items
        #剩下的单词，继续生成树，作为items［0］的子节点
        updateTree(items[1::], inTree.children[key], headerTable, count)


def updateHeader(nodeToTest, targetNode):  # this version does not use recursion
    while (nodeToTest.nodeLink != None):  # Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):  # ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

# rootNode=treeNode('root',9,None)
# eye=treeNode('eye',12,rootNode)
# eye1=treeNode('eye1',13,eye)
# eye2=treeNode('eye2',14,eye)
# eye3=treeNode('eye3',16,eye)

# eye2_1=treeNode('eye2_1',15,eye2)
# eye2_2=treeNode('eye2_2',15,eye2)

# eye2.children['eye2_1']=eye2_1
# eye2.children['eye2_2']=eye2_2
# eye.children['e1']=eye1
# eye.children['e2']=eye2
# eye.children['e3']=eye3
# rootNode.children['eye']=eye
#
# rootNode.disp()

# cc=rootNode.children
# print(cc)

simpData=loadSimpDat()
initSet=createInitSet(simpData)
mytree,myHeaderTab=createTree(initSet,3)
print(findPrefixPath('x',myHeaderTab['x'][1]))
