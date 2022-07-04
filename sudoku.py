# cSpell:Words dtype vectorize isin otypes frompyfunc nout
import re
import sys

import numpy as np

import NumpyUtility as NU


class Sudoku:
  reReplace = re.compile("[\[\]]")
  reArray = re.compile("\| ([\d ]) ([\d ]) ([\d ])")
  Element = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9), dtype=int)
  BlockIndices = [[y, x] for y in range(3) for x in range(3)]
  ElementIndices = [[y, x, j, i] for y in range(3) for x in range(3) for j in range(3) for i in range(3)]
  getLen = np.frompyfunc(len, nin=1, nout=1)
  Pop = np.vectorize(lambda x: x.pop(), otypes=[int])
  uAdd = np.frompyfunc(set.union, nin=2, nout=1)
  uSub = np.frompyfunc(set.difference, nin=2, nout=1)
  uSubPair = np.vectorize(lambda x, p: x if x == p else x - p, otypes=[set], excluded=["p"])
  Types = ["block", "column", "row"]
  Index = [0, 1, 2]

  def __init__(self, puzzle=""):
    self.puzzle = np.zeros((3, 3, 3, 3), dtype=int)
    self.candidate = np.full((3, 3, 3, 3), set(), dtype=set)
    self.stringToPuzzle(puzzle)
    self.count = self.getCount  # max 81
    self.updated = True

  def getCount(self):
    return len(self.puzzle[self.puzzle > 0])

  def printPuzzle(self):
    print(self.getCount())
    print("-------------------------")
    for column in self.puzzle:
      output = ["", "", "", ""]  # joinのために1つ余分に用意する
      for row in column:
        for i, blockColumn in enumerate(row):
          output[i] += "| " + Sudoku.reReplace.sub("", str(blockColumn).replace("0", " ")) + " "
      print("|\n".join(output), end="")
      print("-------------------------")
    print("push any key:")
    text = input()
    if text == "end" or text == "e":
      print(self.candidate)
      sys.exit(0)

  def printBlock(self, y, x):
    print(Sudoku.reReplace.sub(" ", str(self.puzzle[y, x])))
    # return [[v for x in range(3)] for y in range(3)]

  @staticmethod
  def getBlock(array, y, x):
    return array[y, x]

  @staticmethod
  def getColumn(array, y, j):
    return array[y, :, j, :]

  @staticmethod
  def getRow(array, x, i):
    return array[:, x, :, i]

  @staticmethod
  def check(block):  # 完成したかのチェック
    if block.sum() != 45:
      return False
    work = np.array([len(block[block == i]) for i in range(1, 10)])
    if len(work[work > 1]):
      return False
    return True

  def checkAll(self):
    if self.updated == True:
      return
    for index in Sudoku.BlockIndices:
      if not self.checkBlock(*index):
        self.done = False
        return
    for y, x, j, i in Sudoku.ElementIndices:
      if not self.checkColumn(y, j) and self.checkRow(x, i):
        self.done = False
        return
    self.done = True
    return

  def checkBlock(self, y, x):
    return self.check(Sudoku.getBlock(self.puzzle, y, x))

  def checkColumn(self, y, j):
    return self.check(Sudoku.getColumn(self.puzzle, y, j))

  def checkRow(self, x, i):
    return self.check(Sudoku.getRow(self.puzzle, x, i))

  def getCandidate(self, y, x, j, i):  # 候補を探す
    # print(f"getCandidate: {y}, {x}, {j}, {i}: {self.puzzle[y, x, j, i]}")
    if self.puzzle[y, x, j, i] != 0:
      self.candidate[y, x, j, i] = set()
      return
    block = Sudoku.getBlock(self.puzzle, y, x)
    column = Sudoku.getColumn(self.puzzle, y, j)
    row = Sudoku.getRow(self.puzzle, x, i)
    # NU.npInfo(block)
    # NU.npInfo(column)
    # NU.npInfo(row)
    candidate = np.unique(np.concatenate((block[block != 0], column[column != 0], row[row != 0])))
    candidate = ~np.isin(Sudoku.Element, candidate)
    self.candidate[y, x, j, i] = set(Sudoku.Element[candidate])
    print(f"getCandidate: {y}, {x}, {j}, {i}: {self.candidate[y, x, j, i]}")

  def getAllCandidate(self):
    for index in Sudoku.ElementIndices:
      self.getCandidate(*index)
    print(f"getAllCandidate:\n{self.candidate}")

  def perform(self):
    while True:
      self.update()
      self.checkAllPair()
      self.checkAllUnique()
      self.checkAllLine()
      self.checkAll()
      if not self.updated and self.done:
        print("done.")
        return
      elif not self.updated and not self.done:
        print("fail.")
        print(self.candidate)
        return

  def update(self):  # 候補から確定したものをpuzzleに反映させる
    while self.updated:
      self.printPuzzle()
      self.getAllCandidate()
      self.updatePuzzle()

  def updatePuzzle(self):  # crbe. 候補がそれ1つしかない場合は、puzzleを更新する
    l = Sudoku.getLen(self.candidate)
    l = np.nonzero(l == 1)
    if len(self.candidate[l]) == 0:
      self.updated = False
      return
    # print(f"update")
    # print(self.candidate)
    self.puzzle[l] = Sudoku.Pop(self.candidate[l])
    self.updated = True

  @staticmethod
  def getIndex(y, x):
    return y * 3 + x

  def getUniqueCandidate(self, block, i):
    # data = np.concatenate((block[b:b + 1], block[0:b], block[b + 1:], column[0:c], column[c + 1:], row[0:r], row[r + 1:]))
    data = np.concatenate((block[i:i + 1], block[0:i], block[i + 1:]))
    data = Sudoku.uSub.reduce(data)
    if len(data) == 1:
      return data.pop()
    return 0

  def checkUnique(self, y, x, j, i, type):  # 他のマスに存在しない候補を見つけてpuzzleを更新する
    if type == "column":
      block = Sudoku.getColumn(self.candidate, y, j).flatten()
      index = Sudoku.getIndex(x, i)
    elif type == "row":
      block = Sudoku.getRow(self.candidate, x, i).flatten()
      index = Sudoku.getIndex(y, j)
    else:
      block = Sudoku.getBlock(self.candidate, y, x).flatten()
      index = Sudoku.getIndex(j, i)
    result = self.getUniqueCandidate(block, index)
    # print(f"{type}: {y}, {x}, {j}, {i}: {block}, {result}")
    if result != 0:
      self.puzzle[y, x, j, i] = result
      return True
    return False

  def checkAllUnique(self):
    # if self.updated == True:
    #   return
    for index in Sudoku.ElementIndices:
      for type in Sudoku.Types:
        if self.checkUnique(*index, type):
          self.updated = True
    print(f"checkAllUnique:\n{self.candidate}")

  def checkLineCandidate(self, y, x, j, i):
    # if self.updated == True:
    #   return
    flag = False
    candidate = self.candidate[y, x, j, i]
    for b in Sudoku.Index[:y] + Sudoku.Index[y + 1:]:
      block = Sudoku.getBlock(self.candidate, b, x)
      line = block[:, i]
      other = np.concatenate((block[:, :i], block[:, i + 1:]), axis=1).flatten()
      line = Sudoku.uAdd.reduce(line) - Sudoku.uAdd.reduce(other)
      candidate -= line
      if len(line) > 0:
        # print(y, x, j, i, line, candidate)
        flag = True
    self.candidate[y, x, j, i] = candidate
    for b in Sudoku.Index[:x] + Sudoku.Index[x + 1:]:
      block = Sudoku.getBlock(self.candidate, y, b)
      line = block[j, :]
      other = np.concatenate((block[:j, :], block[j + 1:, :])).flatten()
      line = Sudoku.uAdd.reduce(line) - Sudoku.uAdd.reduce(other)
      candidate -= line
      if len(line) > 0:
        # print(y, x, j, i, line, candidate)
        flag = True
    self.candidate[y, x, j, i] = candidate
    print(y, x, j, i, self.candidate[y, x, j, i], flag)
    return flag

  def checkAllLine(self):
    if self.updated == True:
      return
    for index in Sudoku.ElementIndices:
      if self.checkLineCandidate(*index):
        self.updatePuzzle()
    print(f"checkAllLine:\n{self.candidate}")

  def checkPair(self, y, x, j, i, n):
    block = Sudoku.getBlock(self.candidate, y, x)
    if block[j, i] == set() or len(block[j, i]) != n:
      return
    rj, ri = np.nonzero(block == block[j, i])
    pairs = list(zip(rj, ri))
    if len(pairs) == n:
      # print(f"{y}, {x}, {j}, {i}: block: {block[j, i]}, pairs: {pairs}")
      self.candidate[y, x] = Sudoku.uSubPair(block, p=block[pairs[0]])
      self.updated = True
      # print(self.candidate)
    return

  def checkAllPair(self):
    for index in Sudoku.ElementIndices:
      self.checkPair(*index, 3)
      self.checkPair(*index, 2)
    print(f"checkAllPair:\n{self.candidate}")

  ################################################################################
  ## input
  ################################################################################

  @staticmethod
  def stringToList(string):
    s = string.split("\n")
    result = []
    for l in s:
      if len(l) == 0 or l[0] == "-":
        continue
      matches = Sudoku.reArray.findall(l)
      for match in matches:
        result.append([0 if x == " " else int(x) for x in match])
    return result

  @staticmethod
  def listToArray(lt):
    return np.asarray([lt[9 * y + x + 3 * j] for y in range(3) for x in range(3) for j in range(3)], dtype=int).reshape(3, 3, 3, 3)

  def stringToPuzzle(self, string):
    if string == "":
      return
    self.puzzle = Sudoku.listToArray(Sudoku.stringToList(string))


puzzle = """\
-------------------------
|   5   | 9     |   3 7 |
|     9 | 8 7 3 |     5 |
|   3 7 | 4     | 9 8 2 |
-------------------------
|       |   6 4 |     3 |
| 5 7 6 | 3 8 9 |     4 |
| 3   4 |     7 | 8 6 9 |
-------------------------
|   8 5 | 7 3   |     1 |
|     3 |   4   |       |
|     2 |   9   | 3     |
-------------------------
"""

sudoku = Sudoku(puzzle)
sudoku.perform()
