import unittest
import numpy as np
import pandas as pd
from export_common_significant_motifs import select_common_significant

class SelectionTests(unittest.TestCase):
    def test_both_p_values_strictly_below_threshold_without_median_requirement(self):
        frame=pd.DataFrame({'bz_p_value_rank_binary':[.01,.05,.01,np.nan,np.inf,.03],
                            'metabric_p_value_rank_binary':[.02,.01,.05,.01,.01,.04],
                            'bz_median_patient_rank_score':[0]*6,
                            'metabric_median_patient_rank_score':[0]*6})
        self.assertEqual(select_common_significant(frame).index.tolist(),[0,5])

    def test_direction_is_recomputed_as_boolean_after_filtering_missing_values(self):
        frame=pd.DataFrame({'bz_p_value_rank_binary':[.01,.02,.03],
                            'metabric_p_value_rank_binary':[.02,.03,.01],
                            'bz_hazard_ratio_rank_binary':[2,.5,2],
                            'metabric_hazard_ratio_rank_binary':[3,2,.5],
                            'same_hazard_direction':pd.Series([True,False,False],dtype=object)})
        result=select_common_significant(frame)
        self.assertEqual(result.same_hazard_direction.dtype,bool)
        self.assertEqual(int((~result.same_hazard_direction).sum()),2)

if __name__=='__main__':unittest.main()
