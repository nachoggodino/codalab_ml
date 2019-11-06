import fasttext
import utils
import tweet_preprocessing
import swifter
import hunspell
import spacy
import pandas as pd

def print_results(N, precision, recall, phase=''):
    print("{}: F1-SCORE\t---------->>>>>>>\t{:.3f}".format(phase, 2 * (precision * recall) / (precision + recall)))


def lemmatize_sentence(sentence):
    data = utils.untokenize_sentence(sentence)
    return [token.lemma_ for token in lemmatizer(data)]


if __name__ == '__main__':

    MODEL_NAME = 'ingeotec_model_2'
    bStoreFiles = True
    bTrainModel = True
    bSaveModel = False
    bLoadModel = False

    bPreprocess = True
    bLibreOffice = False
    bLemmatize = False
    bTokenize = False
    bUpsampling = True
    bTrainDev = False

    if bLibreOffice:
        print("Loading Hunspell directory")
        dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")

    if bLemmatize:
        print("Loading Spacy Model")
        lemmatizer = spacy.load("es_core_news_md")  # GLOBAL to avoid loading the model several times

    for sLang in ['es', 'cr', 'mx', 'pe', 'uy']:

        print("Training on -{}-".format(sLang.upper()))

        train_data, dev_data, test_data, _ = utils.read_files(sLang)

        if bTrainDev:
            train_data = pd.concat([train_data, dev_data], ignore_index=True).reset_index(drop=True)

        if bUpsampling:
            print('Upsampling the data...')
            train_data = utils.perform_upsampling(train_data)

        if bPreprocess:
            train_data['content'] = tweet_preprocessing.preprocess(train_data['content'], bLowercasing=True, bPunctuation=True)
            dev_data['content'] = tweet_preprocessing.preprocess(dev_data['content'], bLowercasing=True, bPunctuation=True)
            test_data['content'] = tweet_preprocessing.preprocess(test_data['content'], bLowercasing=True, bPunctuation=True)

        if bTokenize:
            print("Tokenizing...")
            train_data['content'] = train_data.swifter.progress_bar(False).apply(
                lambda row: utils.tokenize_sentence(row.content), axis=1)
            dev_data['content'] = dev_data.swifter.progress_bar(False).apply(
                lambda row: utils.tokenize_sentence(row.content), axis=1)
            test_data['content'] = test_data.swifter.progress_bar(False).apply(
                    lambda row: utils.tokenize_sentence(row.content), axis=1)

        if bLibreOffice:
            print("LibreOffice Processing... ")
            train_data['content'] = train_data.swifter.progress_bar(True).apply(
                lambda row: utils.libreoffice_processing(row.content, dictionary), axis=1)
            dev_data['content'] = dev_data.swifter.apply(
                lambda row: utils.libreoffice_processing(row.content, dictionary), axis=1)
            test_data['content'] = test_data.swifter.apply(
                    lambda row: utils.libreoffice_processing(row.content, dictionary), axis=1)

        if bLemmatize:
            print("Lemmatizing data...")
            train_data['content'] = train_data.swifter.apply(lambda row: lemmatize_sentence(row.content), axis=1)
            dev_data['content'] = dev_data.swifter.apply(lambda row: lemmatize_sentence(row.content), axis=1)
            test_data['content'] = test_data.swifter.apply(lambda row: lemmatize_sentence(row.content), axis=1)

        if bTokenize:
            train_data['content'] = [utils.untokenize_sentence(sentence) for sentence in train_data['content']]
            dev_data['content'] = [utils.untokenize_sentence(sentence) for sentence in dev_data['content']]
            test_data['content'] = [utils.untokenize_sentence(sentence) for sentence in test_data['content']]

        if bStoreFiles:
            utils.csv2ftx(train_data.content, train_data.sentiment, sLang, 'train', 'ftx')
            utils.csv2ftx(dev_data.content, dev_data.sentiment, sLang, 'dev', 'ftx')
            utils.csv2ftx(test_data.content, test_data.sentiment, sLang, 'test', 'ftx')



        if bTrainModel:
            model = fasttext.train_supervised(input='./dataset/ftx/intertass_{}_train.txt'.format(sLang),
                                              pretrained_vectors='./embeddings/ingeotec_embeddings/es-{}-100d/es-{}-100d.vec'
                                              .format(sLang.upper(), sLang.upper()),
                                              lr=0.05, epoch=5, wordNgrams=2, seed=1234)

        if bSaveModel:
            model.save_model(path='./fasttext/models/{}_{}'.format(sLang, MODEL_NAME))

        elif bLoadModel:
            model = fasttext.load_model(path='./fasttext/models/{}_{}'.format(sLang, MODEL_NAME))


        # print_results(*model.test(path='./dataset/ftx/intertass_{}_dev.txt'.format(sLang)), phase='DEV')
        # print_results(*model.test(path='./dataset/ftx/intertass_{}_test.txt'.format(sLang)), phase='TEST')

        # neg_prob, neu_prob, non_prob, pos_prob = [], [], [], []
        #
        # for tweet in dev_data['content']:
        #     result = model.predict(tweet, k=4)
        #     for label, prob in zip(result[0], result[1]):
        #         if label[-1] is '0':
        #             neg_prob.append(prob)
        #         if label[-1] is '1':
        #             neu_prob.append(prob)
        #         if label[-1] is '2':
        #             non_prob.append(prob)
        #         if label[-1] is '3':
        #             pos_prob.append(prob)
        #
        # dev_probs_df = pd.DataFrame({
        #     'N': neg_prob,
        #     'NEU': neu_prob,
        #     'NONE': non_prob,
        #     'P': pos_prob,
        # })
        #
        # dev_probs_df.to_csv('./fasttext/probabilities/{}_{}_dev.csv'.format(sLang, MODEL_NAME), encoding='utf-8', sep='\t')
        #
        # neg_prob, neu_prob, non_prob, pos_prob = [], [], [], []
        #
        # for tweet in test_data['content']:
        #     result = model.predict(tweet, k=4)
        #     for label, prob in zip(result[0], result[1]):
        #         if label[-1] is '0':
        #             neg_prob.append(prob)
        #         if label[-1] is '1':
        #             neu_prob.append(prob)
        #         if label[-1] is '2':
        #             non_prob.append(prob)
        #         if label[-1] is '3':
        #             pos_prob.append(prob)
        #
        # test_probs_df = pd.DataFrame({
        #     'N': neg_prob,
        #     'NEU': neu_prob,
        #     'NONE': non_prob,
        #     'P': pos_prob,
        # })
        # test_probs_df.to_csv('./fasttext/probabilities/{}_{}_test.csv'.format(sLang, MODEL_NAME), encoding='utf-8', sep='\t')

        predictions = [int(model.predict(tweet)[0][0][-1]) for tweet in dev_data['content']]
        print('DEV')
        utils.print_confusion_matrix(predictions, dev_data.sentiment)
        predictions = [int(model.predict(tweet)[0][0][-1]) for tweet in test_data['content']]
        print('TEST')
        utils.print_confusion_matrix(predictions, test_data.sentiment)


        print('-----------------------------------NEXT---------------------------------------------------')





