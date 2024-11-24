import numpy as np
import leopardgecko.segmentor2 as lgs2

def get_2_12_data_and_labels_vols64():

    data0 = np.random.rand(2,12,3,64,64,64)
    labels = np.random.randint(0,3,size=(2,64,64,64))
    return data0, labels


def test_train_nn2_default(get_2_12_data_and_labels_vols64):

    data,labels = get_2_12_data_and_labels_vols64

    lgs2.nn1_train_epochs=2 # debug low number
    lgs2.nn2_train_epochs=2



    lgs2.nn1_models_class_generator= [lgs2.nn1_dict_gen_default,
        lgs2.nn1_dict_gen_default.copy()]
    
    lgs2.nn1_axes_to_models_indices = [0,1,1]

    lgs2.nn2_MLP_model_class_generator= lgs2.nn2_MLP_model_class_generator_default
    # Default 3 unet models, one per axis. 3 classes
    # NN2, MLP 10,10

    #nn1_train_epochs= 10

    lgs2.update_nn1_models_from_generators()
    lgs2.update_nn2_model_from_generator()

    #run the test here
    lgs2.train_nn2_default(data,labels)

    print("Test complete")

if __name__ == "__main__":
    d = get_2_12_data_and_labels_vols64()
    test_train_nn2_default(d)