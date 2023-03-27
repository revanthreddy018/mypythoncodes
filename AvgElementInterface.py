import abc




class MyAvgInterface(abc.ABC):
    @classmethod
    def GetAverageLastN(cls):
        '''
        This function will calculate moving average of last N elements

        :return: Nothing
        '''
        pass

    @classmethod
    def AddElement(cls):
        '''
        This function will add a single element to List data structure

        :return: Nothing
        '''
        pass

    @classmethod
    def AccessElement(cls):
        '''
        This function will allow to access single element based on
        its position - with 1st element being at position 1 - human readable form
        though index in python begin with 0

        :return: Nothing
        '''
        pass
