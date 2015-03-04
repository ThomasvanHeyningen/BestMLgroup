import pandas as pd
import time

class Tester:

    def __init__(self, model, classnames):
        self.model=model
        self.classnames=classnames


    def test(self, testset, imagefilenames):

        prediction_prob = self.model.predict_proba(testset)
        df = pd.DataFrame(prediction_prob, columns=self.classnames, index=imagefilenames)
        df.index.name = 'image'
        filename = '../submission%s.csv' %(time.strftime("%Y%m%d%H%M%S"))
        print "saving submission"
        df.to_csv(filename)