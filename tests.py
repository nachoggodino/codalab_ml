import pandas


def fasttext_to_df(fasttext_path):
    with open('./cr_dev_mono.out', 'r') as ft_file:
        negs, poss, neus, nones = [], [], [], []
        for i, line in enumerate(ft_file):
            words = line.split()
            print(str(len(words)))
            for j, word in enumerate(words):
                if j % 2 is 1:
                    continue
                else:
                    label = word.replace('__label__', '')
                    if label == 'N':
                        negs.append(words[j+1])
                        print('neg')
                    if label == 'P':
                        poss.append(words[j+1])
                        print('pos')
                    if label == 'NEU':
                        neus.append(words[j+1])
                        print('neu')
                    if label == 'NONE':
                        nones.append(words[j+1])
                        print('none')

        print(str(len(negs)))
        print(str(len(poss)))

        file_df = pandas.DataFrame({
                        'N': negs,
                        'NEU': neus,
                        'NONE': nones,
                        'P': poss,
        })

print(file_df)
