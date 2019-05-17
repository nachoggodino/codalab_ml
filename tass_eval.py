import re

def evalTask1(qrel_file, run_file):

  # Load QREL file
  QREL = {}
  with open(qrel_file) as fd:
    for line in fd:
      line = line.strip()
      if re.match('^(.+?)\t(.+?)$',line):
        (id, value) = line.split('\t')
        QREL[id] = value

  # Load RUN file
  DATA = {}
  with open(run_file) as fd:
    for line in fd:
      line = line.strip()
      if re.match('^(.+?)\t(.+?)$',line):
        (id, value) = line.split('\t')
        if id in QREL:
          DATA[id] = value
        else:
          print('Ignoring unknown instance id: {}'.format(id))
      else:
        print('Wrong run file format: {}'.format(line))
        return False

  # Calculate Confusion Matrix, True Positives, False Positives and False Negatives
  CONFUSION = {}
  aTP = {}
  aFP = {}
  aFN = {}
  for (id,valueqrel) in QREL.items():
    value = DATA.get(id,'?')
    if valueqrel == value:
      aTP[valueqrel] = aTP.get(valueqrel,0) + 1
    else:
      aFP[value] = aFP.get(value,0) + 1
      aFN[valueqrel] = aFN.get(valueqrel,0) + 1

    if not valueqrel in CONFUSION: 
      CONFUSION[valueqrel] = {value: 1}
    else:
      CONFUSION[valueqrel][value] = CONFUSION[valueqrel].get(value,0) + 1

  tp = 0
  fp = 0
  fn = 0
  for valueqrel in CONFUSION:
    tp += aTP.get(valueqrel,0)
    fp += aFP.get(valueqrel,0)
    fn += aFN.get(valueqrel,0)

  a = tp/len(QREL)
  mip = tp/(tp+fp)
  mir = tp/(tp+fn)
  if mip+mir==0:
    mif1 = 0
  else:
    mif1 = 2*mip*mir/(mip+mir)

  aP = {}
  aR = {}
  aF1 = {}
  map_value = 0
  mar = 0
  maf1 = 0
  for valueqrel in CONFUSION:
    try:
      aP[valueqrel] = 0 if aTP[valueqrel]+aFP[valueqrel]==0 else aTP[valueqrel]/(aTP[valueqrel]+aFP[valueqrel])
    except:
      aP[valueqrel] = 0
    try:
      aR[valueqrel] = 0 if aTP[valueqrel]+aFN[valueqrel]==0 else aTP[valueqrel]/(aTP[valueqrel]+aFN[valueqrel])
    except:
      aR[valueqrel] = 0
    try:
      aF1[valueqrel] = 0 if aP[valueqrel]+aR[valueqrel]==0 else  2*aP[valueqrel]*aR[valueqrel]/(aP[valueqrel]+aR[valueqrel])
    except:
      aF1[valueqrel] = 0
      
    map_value += aP[valueqrel]
    mar += aR[valueqrel]

  map_value /= len(CONFUSION)
  mar /= len(CONFUSION)
  maf1 = 0 if map_value+mar==0 else 2*map_value*mar/(map_value+mar)

  return {'maf1': maf1, 'map': map_value, 'mar': mar, 'a': a}
