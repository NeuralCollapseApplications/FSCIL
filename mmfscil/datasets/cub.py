import os
import os.path as osp
from collections import defaultdict
from typing import List, Dict, Mapping, Optional, Tuple
from typing_extensions import Literal

import copy
import mmcv
import numpy as np
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.pipelines import Compose

from torch.utils.data import Dataset

CLASSES = [
    '001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', '004.Groove_billed_Ani',
    '005.Crested_Auklet', '006.Least_Auklet', '007.Parakeet_Auklet', '008.Rhinoceros_Auklet', '009.Brewer_Blackbird',

    '010.Red_winged_Blackbird', '011.Rusty_Blackbird', '012.Yellow_headed_Blackbird', '013.Bobolink',
    '014.Indigo_Bunting', '015.Lazuli_Bunting', '016.Painted_Bunting', '017.Cardinal', '018.Spotted_Catbird',
    '019.Gray_Catbird', '020.Yellow_breasted_Chat',

    '021.Eastern_Towhee', '022.Chuck_will_Widow', '023.Brandt_Cormorant', '024.Red_faced_Cormorant',
    '025.Pelagic_Cormorant', '026.Bronzed_Cowbird', '027.Shiny_Cowbird', '028.Brown_Creeper', '029.American_Crow',
    '030.Fish_Crow',

    '031.Black_billed_Cuckoo', '032.Mangrove_Cuckoo', '033.Yellow_billed_Cuckoo', '034.Gray_crowned_Rosy_Finch',
    '035.Purple_Finch', '036.Northern_Flicker', '037.Acadian_Flycatcher', '038.Great_Crested_Flycatcher',
    '039.Least_Flycatcher', '040.Olive_sided_Flycatcher',

    '041.Scissor_tailed_Flycatcher', '042.Vermilion_Flycatcher', '043.Yellow_bellied_Flycatcher', '044.Frigatebird',
    '045.Northern_Fulmar',
    '046.Gadwall', '047.American_Goldfinch', '048.European_Goldfinch', '049.Boat_tailed_Grackle', '050.Eared_Grebe',

    '051.Horned_Grebe', '052.Pied_billed_Grebe', '053.Western_Grebe', '054.Blue_Grosbeak', '055.Evening_Grosbeak',
    '056.Pine_Grosbeak', '057.Rose_breasted_Grosbeak', '058.Pigeon_Guillemot', '059.California_Gull',
    '060.Glaucous_winged_Gull',

    '061.Heermann_Gull', '062.Herring_Gull', '063.Ivory_Gull', '064.Ring_billed_Gull', '065.Slaty_backed_Gull',
    '066.Western_Gull', '067.Anna_Hummingbird', '068.Ruby_throated_Hummingbird', '069.Rufous_Hummingbird',
    '070.Green_Violetear',

    '071.Long_tailed_Jaeger', '072.Pomarine_Jaeger', '073.Blue_Jay', '074.Florida_Jay', '075.Green_Jay',
    '076.Dark_eyed_Junco', '077.Tropical_Kingbird', '078.Gray_Kingbird', '079.Belted_Kingfisher',
    '080.Green_Kingfisher',

    '081.Pied_Kingfisher', '082.Ringed_Kingfisher', '083.White_breasted_Kingfisher', '084.Red_legged_Kittiwake',
    '085.Horned_Lark',
    '086.Pacific_Loon', '087.Mallard', '088.Western_Meadowlark', '089.Hooded_Merganser', '090.Red_breasted_Merganser',

    '091.Mockingbird', '092.Nighthawk', '093.Clark_Nutcracker', '094.White_breasted_Nuthatch', '095.Baltimore_Oriole',
    '096.Hooded_Oriole', '097.Orchard_Oriole', '098.Scott_Oriole', '099.Ovenbird', '100.Brown_Pelican',

    '101.White_Pelican', '102.Western_Wood_Pewee', '103.Sayornis', '104.American_Pipit', '105.Whip_poor_Will',
    '106.Horned_Puffin', '107.Common_Raven', '108.White_necked_Raven', '109.American_Redstart', '110.Geococcyx',

    '111.Loggerhead_Shrike', '112.Great_Grey_Shrike', '113.Baird_Sparrow', '114.Black_throated_Sparrow',
    '115.Brewer_Sparrow', '116.Chipping_Sparrow', '117.Clay_colored_Sparrow', '118.House_Sparrow', '119.Field_Sparrow',
    '120.Fox_Sparrow',

    '121.Grasshopper_Sparrow', '122.Harris_Sparrow', '123.Henslow_Sparrow', '124.Le_Conte_Sparrow',
    '125.Lincoln_Sparrow', '126.Nelson_Sharp_tailed_Sparrow', '127.Savannah_Sparrow', '128.Seaside_Sparrow',
    '129.Song_Sparrow', '130.Tree_Sparrow',

    '131.Vesper_Sparrow', '132.White_crowned_Sparrow', '133.White_throated_Sparrow', '134.Cape_Glossy_Starling',
    '135.Bank_Swallow', '136.Barn_Swallow', '137.Cliff_Swallow', '138.Tree_Swallow', '139.Scarlet_Tanager',
    '140.Summer_Tanager',

    '141.Artic_Tern', '142.Black_Tern', '143.Caspian_Tern', '144.Common_Tern', '145.Elegant_Tern',
    '146.Forsters_Tern', '147.Least_Tern', '148.Green_tailed_Towhee', '149.Brown_Thrasher', '150.Sage_Thrasher',

    '151.Black_capped_Vireo', '152.Blue_headed_Vireo', '153.Philadelphia_Vireo', '154.Red_eyed_Vireo',
    '155.Warbling_Vireo',
    '156.White_eyed_Vireo', '157.Yellow_throated_Vireo', '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler',
    '160.Black_throated_Blue_Warbler',

    '161.Blue_winged_Warbler', '162.Canada_Warbler', '163.Cape_May_Warbler', '164.Cerulean_Warbler',
    '165.Chestnut_sided_Warbler', '166.Golden_winged_Warbler', '167.Hooded_Warbler', '168.Kentucky_Warbler',
    '169.Magnolia_Warbler',
    '170.Mourning_Warbler',

    '171.Myrtle_Warbler', '172.Nashville_Warbler', '173.Orange_crowned_Warbler', '174.Palm_Warbler', '175.Pine_Warbler',
    '176.Prairie_Warbler', '177.Prothonotary_Warbler', '178.Swainson_Warbler', '179.Tennessee_Warbler',
    '180.Wilson_Warbler',

    '181.Worm_eating_Warbler', '182.Yellow_Warbler', '183.Northern_Waterthrush', '184.Louisiana_Waterthrush',
    '185.Bohemian_Waxwing', '186.Cedar_Waxwing', '187.American_Three_toed_Woodpecker', '188.Pileated_Woodpecker',
    '189.Red_bellied_Woodpecker', '190.Red_cockaded_Woodpecker',

    '191.Red_headed_Woodpecker', '192.Downy_Woodpecker', '193.Bewick_Wren', '194.Cactus_Wren', '195.Carolina_Wren',
    '196.House_Wren', '197.Marsh_Wren', '198.Rock_Wren', '199.Winter_Wren', '200.Common_Yellowthroat'
]

# Please refer to https://github.com/icoz69/CEC-CVPR2021/tree/dc26237/data/index_list/cub200
FSCIL_SAMPLES = {
    '101.White_Pelican': ['White_Pelican_0081_96148.jpg', 'White_Pelican_0075_96422.jpg',
                          'White_Pelican_0026_95832.jpg', 'White_Pelican_0022_95897.jpg',
                          'White_Pelican_0044_96028.jpg'],
    '102.Western_Wood_Pewee': ['Western_Wood_Pewee_0072_98035.jpg', 'Western_Wood_Pewee_0004_98257.jpg',
                               'Western_Wood_Pewee_0060_795045.jpg', 'Western_Wood_Pewee_0039_795063.jpg',
                               'Western_Wood_Pewee_0040_795051.jpg'],
    '103.Sayornis': ['Sayornis_0099_98593.jpg', 'Sayornis_0133_99129.jpg', 'Sayornis_0098_98419.jpg',
                     'Sayornis_0011_98610.jpg', 'Sayornis_0114_98976.jpg'],
    '104.American_Pipit': ['American_Pipit_0037_99954.jpg', 'American_Pipit_0067_100237.jpg',
                           'American_Pipit_0019_99810.jpg', 'American_Pipit_0058_100218.jpg',
                           'American_Pipit_0113_99939.jpg'],
    '105.Whip_poor_Will': ['Whip_Poor_Will_0038_100443.jpg', 'Whip_Poor_Will_0018_796403.jpg',
                           'Whip_Poor_Will_0013_796439.jpg', 'Whip_Poor_Will_0026_100456.jpg',
                           'Whip_Poor_Will_0004_100479.jpg'],
    '106.Horned_Puffin': ['Horned_Puffin_0004_100733.jpg', 'Horned_Puffin_0028_100765.jpg',
                          'Horned_Puffin_0062_100693.jpg', 'Horned_Puffin_0042_100760.jpg',
                          'Horned_Puffin_0030_100725.jpg'],
    '107.Common_Raven': ['Common_Raven_0009_102112.jpg', 'Common_Raven_0068_101216.jpg', 'Common_Raven_0099_102534.jpg',
                         'Common_Raven_0001_101213.jpg', 'Common_Raven_0095_101831.jpg'],
    '108.White_necked_Raven': ['White_Necked_Raven_0063_797361.jpg', 'White_Necked_Raven_0050_797374.jpg',
                               'White_Necked_Raven_0010_797350.jpg', 'White_Necked_Raven_0002_797370.jpg',
                               'White_Necked_Raven_0026_797357.jpg'],
    '109.American_Redstart': ['American_Redstart_0036_103231.jpg', 'American_Redstart_0071_103266.jpg',
                              'American_Redstart_0085_103155.jpg', 'American_Redstart_0056_103241.jpg',
                              'American_Redstart_0049_103176.jpg'],
    '110.Geococcyx': ['Geococcyx_0106_104216.jpg', 'Geococcyx_0086_104755.jpg', 'Geococcyx_0124_104141.jpg',
                      'Geococcyx_0117_104227.jpg', 'Geococcyx_0036_104173.jpg'],
    '111.Loggerhead_Shrike': ['Loggerhead_Shrike_0127_105742.jpg', 'Loggerhead_Shrike_0018_26407.jpg',
                              'Loggerhead_Shrike_0019_106132.jpg', 'Loggerhead_Shrike_0011_104921.jpg',
                              'Loggerhead_Shrike_0033_105686.jpg'],
    '112.Great_Grey_Shrike': ['Great_Grey_Shrike_0092_797048.jpg', 'Great_Grey_Shrike_0042_797056.jpg',
                              'Great_Grey_Shrike_0049_797025.jpg', 'Great_Grey_Shrike_0083_797051.jpg',
                              'Great_Grey_Shrike_0063_797042.jpg'],
    '113.Baird_Sparrow': ['Baird_Sparrow_0021_794576.jpg', 'Baird_Sparrow_0018_794584.jpg',
                          'Baird_Sparrow_0025_794564.jpg', 'Baird_Sparrow_0041_794582.jpg',
                          'Baird_Sparrow_0036_794572.jpg'],
    '114.Black_throated_Sparrow': ['Black_Throated_Sparrow_0019_107192.jpg', 'Black_Throated_Sparrow_0088_107220.jpg',
                                   'Black_Throated_Sparrow_0097_106935.jpg', 'Black_Throated_Sparrow_0055_107213.jpg',
                                   'Black_Throated_Sparrow_0010_107375.jpg'],
    '115.Brewer_Sparrow': ['Brewer_Sparrow_0068_107422.jpg', 'Brewer_Sparrow_0036_107451.jpg',
                           'Brewer_Sparrow_0041_796711.jpg', 'Brewer_Sparrow_0014_107435.jpg',
                           'Brewer_Sparrow_0076_107393.jpg'],
    '116.Chipping_Sparrow': ['Chipping_Sparrow_0064_108204.jpg', 'Chipping_Sparrow_0038_109234.jpg',
                             'Chipping_Sparrow_0098_108644.jpg', 'Chipping_Sparrow_0110_108974.jpg',
                             'Chipping_Sparrow_0023_108684.jpg'],
    '117.Clay_colored_Sparrow': ['Clay_Colored_Sparrow_0104_110699.jpg', 'Clay_Colored_Sparrow_0098_110735.jpg',
                                 'Clay_Colored_Sparrow_0003_110672.jpg', 'Clay_Colored_Sparrow_0029_110720.jpg',
                                 'Clay_Colored_Sparrow_0087_110946.jpg'],
    '118.House_Sparrow': ['House_Sparrow_0092_111413.jpg', 'House_Sparrow_0111_112968.jpg',
                          'House_Sparrow_0080_111099.jpg', 'House_Sparrow_0130_110985.jpg',
                          'House_Sparrow_0053_111388.jpg'],
    '119.Field_Sparrow': ['Field_Sparrow_0069_113827.jpg', 'Field_Sparrow_0130_113846.jpg',
                          'Field_Sparrow_0091_113486.jpg', 'Field_Sparrow_0043_113607.jpg',
                          'Field_Sparrow_0108_114154.jpg'],
    '120.Fox_Sparrow': ['Fox_Sparrow_0104_114908.jpg', 'Fox_Sparrow_0086_115484.jpg', 'Fox_Sparrow_0055_114809.jpg',
                        'Fox_Sparrow_0012_115324.jpg', 'Fox_Sparrow_0035_114866.jpg'],
    '121.Grasshopper_Sparrow': ['Grasshopper_Sparrow_0014_116129.jpg', 'Grasshopper_Sparrow_0114_116160.jpg',
                                'Grasshopper_Sparrow_0068_115799.jpg', 'Grasshopper_Sparrow_0110_115644.jpg',
                                'Grasshopper_Sparrow_0042_115638.jpg'],
    '122.Harris_Sparrow': ['Harris_Sparrow_0006_116364.jpg', 'Harris_Sparrow_0018_116402.jpg',
                           'Harris_Sparrow_0026_116620.jpg', 'Harris_Sparrow_0020_116379.jpg',
                           'Harris_Sparrow_0011_116597.jpg'],
    '123.Henslow_Sparrow': ['Henslow_Sparrow_0023_796582.jpg', 'Henslow_Sparrow_0052_796599.jpg',
                            'Henslow_Sparrow_0054_116850.jpg', 'Henslow_Sparrow_0064_796573.jpg',
                            'Henslow_Sparrow_0070_796571.jpg'],
    '124.Le_Conte_Sparrow': ['Le_Conte_Sparrow_0040_117088.jpg', 'Le_Conte_Sparrow_0072_795230.jpg',
                             'Le_Conte_Sparrow_0068_795180.jpg', 'Le_Conte_Sparrow_0081_795215.jpg',
                             'Le_Conte_Sparrow_0032_795186.jpg'],
    '125.Lincoln_Sparrow': ['Lincoln_Sparrow_0084_117492.jpg', 'Lincoln_Sparrow_0009_117535.jpg',
                            'Lincoln_Sparrow_0014_117883.jpg', 'Lincoln_Sparrow_0042_117507.jpg',
                            'Lincoln_Sparrow_0072_117951.jpg'],
    '126.Nelson_Sharp_tailed_Sparrow': ['Nelson_Sharp_Tailed_Sparrow_0056_117974.jpg',
                                        'Nelson_Sharp_Tailed_Sparrow_0002_796908.jpg',
                                        'Nelson_Sharp_Tailed_Sparrow_0051_796902.jpg',
                                        'Nelson_Sharp_Tailed_Sparrow_0014_796906.jpg',
                                        'Nelson_Sharp_Tailed_Sparrow_0077_796913.jpg'],
    '127.Savannah_Sparrow': ['Savannah_Sparrow_0049_119596.jpg', 'Savannah_Sparrow_0118_118603.jpg',
                             'Savannah_Sparrow_0068_119972.jpg', 'Savannah_Sparrow_0052_118583.jpg',
                             'Savannah_Sparrow_0054_120057.jpg'],
    '128.Seaside_Sparrow': ['Seaside_Sparrow_0001_120720.jpg', 'Seaside_Sparrow_0048_120758.jpg',
                            'Seaside_Sparrow_0042_796528.jpg', 'Seaside_Sparrow_0049_120735.jpg',
                            'Seaside_Sparrow_0035_796533.jpg'],
    '129.Song_Sparrow': ['Song_Sparrow_0046_121903.jpg', 'Song_Sparrow_0055_121158.jpg', 'Song_Sparrow_0107_120990.jpg',
                         'Song_Sparrow_0091_121651.jpg', 'Song_Sparrow_0087_121062.jpg'],
    '130.Tree_Sparrow': ['Tree_Sparrow_0094_124974.jpg', 'Tree_Sparrow_0123_125324.jpg', 'Tree_Sparrow_0041_123497.jpg',
                         'Tree_Sparrow_0086_123751.jpg', 'Tree_Sparrow_0119_124114.jpg'],
    '131.Vesper_Sparrow': ['Vesper_Sparrow_0079_125579.jpg', 'Vesper_Sparrow_0080_125606.jpg',
                           'Vesper_Sparrow_0084_125532.jpg', 'Vesper_Sparrow_0094_125602.jpg',
                           'Vesper_Sparrow_0019_125558.jpg'],
    '132.White_crowned_Sparrow': ['White_Crowned_Sparrow_0068_126156.jpg', 'White_Crowned_Sparrow_0100_126267.jpg',
                                  'White_Crowned_Sparrow_0072_127080.jpg', 'White_Crowned_Sparrow_0033_127728.jpg',
                                  'White_Crowned_Sparrow_0095_127118.jpg'],
    '133.White_throated_Sparrow': ['White_Throated_Sparrow_0125_128832.jpg', 'White_Throated_Sparrow_0056_128906.jpg',
                                   'White_Throated_Sparrow_0085_129180.jpg', 'White_Throated_Sparrow_0042_128899.jpg',
                                   'White_Throated_Sparrow_0021_128804.jpg'],
    '134.Cape_Glossy_Starling': ['Cape_Glossy_Starling_0096_129388.jpg', 'Cape_Glossy_Starling_0046_129434.jpg',
                                 'Cape_Glossy_Starling_0043_129358.jpg', 'Cape_Glossy_Starling_0019_129407.jpg',
                                 'Cape_Glossy_Starling_0067_129380.jpg'],
    '135.Bank_Swallow': ['Bank_Swallow_0003_129623.jpg', 'Bank_Swallow_0045_129483.jpg', 'Bank_Swallow_0020_129747.jpg',
                         'Bank_Swallow_0067_129959.jpg', 'Bank_Swallow_0053_129501.jpg'],
    '136.Barn_Swallow': ['Barn_Swallow_0018_130709.jpg', 'Barn_Swallow_0048_132793.jpg', 'Barn_Swallow_0070_130127.jpg',
                         'Barn_Swallow_0066_130214.jpg', 'Barn_Swallow_0049_130181.jpg'],
    '137.Cliff_Swallow': ['Cliff_Swallow_0018_132974.jpg', 'Cliff_Swallow_0023_134314.jpg',
                          'Cliff_Swallow_0066_133206.jpg', 'Cliff_Swallow_0050_134054.jpg',
                          'Cliff_Swallow_0075_134516.jpg'],
    '138.Tree_Swallow': ['Tree_Swallow_0087_137354.jpg', 'Tree_Swallow_0043_136878.jpg', 'Tree_Swallow_0111_135253.jpg',
                         'Tree_Swallow_0108_135068.jpg', 'Tree_Swallow_0064_136322.jpg'],
    '139.Scarlet_Tanager': ['Scarlet_Tanager_0107_138577.jpg', 'Scarlet_Tanager_0077_137626.jpg',
                            'Scarlet_Tanager_0040_137885.jpg', 'Scarlet_Tanager_0033_137603.jpg',
                            'Scarlet_Tanager_0132_138001.jpg'],
    '140.Summer_Tanager': ['Summer_Tanager_0032_140425.jpg', 'Summer_Tanager_0046_139802.jpg',
                           'Summer_Tanager_0111_139605.jpg', 'Summer_Tanager_0116_139923.jpg',
                           'Summer_Tanager_0095_139882.jpg'],
    '141.Artic_Tern': ['Artic_Tern_0055_141524.jpg', 'Artic_Tern_0124_142121.jpg', 'Artic_Tern_0133_141069.jpg',
                       'Artic_Tern_0111_143101.jpg', 'Artic_Tern_0107_141181.jpg'],
    '142.Black_Tern': ['Black_Tern_0079_143998.jpg', 'Black_Tern_0082_144372.jpg', 'Black_Tern_0029_144140.jpg',
                       'Black_Tern_0066_144541.jpg', 'Black_Tern_0046_144229.jpg'],
    '143.Caspian_Tern': ['Caspian_Tern_0009_145057.jpg', 'Caspian_Tern_0116_145607.jpg', 'Caspian_Tern_0123_145774.jpg',
                         'Caspian_Tern_0006_145594.jpg', 'Caspian_Tern_0013_145553.jpg'],
    '144.Common_Tern': ['Common_Tern_0071_148796.jpg', 'Common_Tern_0077_149196.jpg', 'Common_Tern_0030_147825.jpg',
                        'Common_Tern_0095_149960.jpg', 'Common_Tern_0083_148096.jpg'],
    '145.Elegant_Tern': ['Elegant_Tern_0009_150954.jpg', 'Elegant_Tern_0045_150752.jpg', 'Elegant_Tern_0046_150905.jpg',
                         'Elegant_Tern_0103_150493.jpg', 'Elegant_Tern_0004_150948.jpg'],
    '146.Forsters_Tern': ['Forsters_Tern_0027_151456.jpg', 'Forsters_Tern_0077_152255.jpg',
                          'Forsters_Tern_0125_151399.jpg', 'Forsters_Tern_0045_151227.jpg',
                          'Forsters_Tern_0119_152709.jpg'],
    '147.Least_Tern': ['Least_Tern_0092_153361.jpg', 'Least_Tern_0020_153458.jpg', 'Least_Tern_0060_153190.jpg',
                       'Least_Tern_0119_153950.jpg', 'Least_Tern_0037_153637.jpg'],
    '148.Green_tailed_Towhee': ['Green_Tailed_Towhee_0018_154825.jpg', 'Green_Tailed_Towhee_0070_154844.jpg',
                                'Green_Tailed_Towhee_0064_154771.jpg', 'Green_Tailed_Towhee_0058_797399.jpg',
                                'Green_Tailed_Towhee_0060_154820.jpg'],
    '149.Brown_Thrasher': ['Brown_Thrasher_0013_155329.jpg', 'Brown_Thrasher_0079_155394.jpg',
                           'Brown_Thrasher_0019_155216.jpg', 'Brown_Thrasher_0051_155344.jpg',
                           'Brown_Thrasher_0081_155256.jpg'],
    '150.Sage_Thrasher': ['Sage_Thrasher_0033_155511.jpg', 'Sage_Thrasher_0069_155544.jpg',
                          'Sage_Thrasher_0096_155449.jpg', 'Sage_Thrasher_0104_155529.jpg',
                          'Sage_Thrasher_0070_155732.jpg'],
    '151.Black_capped_Vireo': ['Black_Capped_Vireo_0012_797473.jpg', 'Black_Capped_Vireo_0007_797481.jpg',
                               'Black_Capped_Vireo_0020_797461.jpg', 'Black_Capped_Vireo_0053_797478.jpg',
                               'Black_Capped_Vireo_0003_797467.jpg'],
    '152.Blue_headed_Vireo': ['Blue_Headed_Vireo_0097_156272.jpg', 'Blue_Headed_Vireo_0019_156311.jpg',
                              'Blue_Headed_Vireo_0121_156233.jpg', 'Blue_Headed_Vireo_0011_156276.jpg',
                              'Blue_Headed_Vireo_0119_156259.jpg'],
    '153.Philadelphia_Vireo': ['Philadelphia_Vireo_0078_794776.jpg', 'Philadelphia_Vireo_0039_794794.jpg',
                               'Philadelphia_Vireo_0068_794763.jpg', 'Philadelphia_Vireo_0012_794785.jpg',
                               'Philadelphia_Vireo_0013_794772.jpg'],
    '154.Red_eyed_Vireo': ['Red_Eyed_Vireo_0101_156988.jpg', 'Red_Eyed_Vireo_0006_157025.jpg',
                           'Red_Eyed_Vireo_0041_156954.jpg', 'Red_Eyed_Vireo_0115_157004.jpg',
                           'Red_Eyed_Vireo_0056_156968.jpg'],
    '155.Warbling_Vireo': ['Warbling_Vireo_0075_158480.jpg', 'Warbling_Vireo_0061_158494.jpg',
                           'Warbling_Vireo_0004_158376.jpg', 'Warbling_Vireo_0030_158488.jpg',
                           'Warbling_Vireo_0077_158427.jpg'],
    '156.White_eyed_Vireo': ['White_Eyed_Vireo_0042_159012.jpg', 'White_Eyed_Vireo_0033_159079.jpg',
                             'White_Eyed_Vireo_0126_159341.jpg', 'White_Eyed_Vireo_0071_159072.jpg',
                             'White_Eyed_Vireo_0016_158978.jpg'],
    '157.Yellow_throated_Vireo': ['Yellow_Throated_Vireo_0066_795007.jpg', 'Yellow_Throated_Vireo_0032_159632.jpg',
                                  'Yellow_Throated_Vireo_0017_794988.jpg', 'Yellow_Throated_Vireo_0025_795009.jpg',
                                  'Yellow_Throated_Vireo_0058_794994.jpg'],
    '158.Bay_breasted_Warbler': ['Bay_Breasted_Warbler_0073_797138.jpg', 'Bay_Breasted_Warbler_0081_159963.jpg',
                                 'Bay_Breasted_Warbler_0071_797108.jpg', 'Bay_Breasted_Warbler_0105_797143.jpg',
                                 'Bay_Breasted_Warbler_0052_797125.jpg'],
    '159.Black_and_white_Warbler': ['Black_And_White_Warbler_0057_160037.jpg',
                                    'Black_And_White_Warbler_0035_160102.jpg',
                                    'Black_And_White_Warbler_0119_160898.jpg',
                                    'Black_And_White_Warbler_0102_160073.jpg',
                                    'Black_And_White_Warbler_0022_160512.jpg'],
    '160.Black_throated_Blue_Warbler': ['Black_Throated_Blue_Warbler_0050_161154.jpg',
                                        'Black_Throated_Blue_Warbler_0130_161682.jpg',
                                        'Black_Throated_Blue_Warbler_0133_161539.jpg',
                                        'Black_Throated_Blue_Warbler_0054_161158.jpg',
                                        'Black_Throated_Blue_Warbler_0024_161619.jpg'],
    '161.Blue_winged_Warbler': ['Blue_Winged_Warbler_0071_161900.jpg', 'Blue_Winged_Warbler_0035_161741.jpg',
                                'Blue_Winged_Warbler_0054_161862.jpg', 'Blue_Winged_Warbler_0023_161774.jpg',
                                'Blue_Winged_Warbler_0040_161883.jpg'],
    '162.Canada_Warbler': ['Canada_Warbler_0113_162403.jpg', 'Canada_Warbler_0064_162417.jpg',
                           'Canada_Warbler_0091_162378.jpg', 'Canada_Warbler_0016_162411.jpg',
                           'Canada_Warbler_0080_162392.jpg'],
    '163.Cape_May_Warbler': ['Cape_May_Warbler_0012_162701.jpg', 'Cape_May_Warbler_0103_162972.jpg',
                             'Cape_May_Warbler_0022_162912.jpg', 'Cape_May_Warbler_0005_163197.jpg',
                             'Cape_May_Warbler_0032_162659.jpg'],
    '164.Cerulean_Warbler': ['Cerulean_Warbler_0039_163420.jpg', 'Cerulean_Warbler_0020_163353.jpg',
                             'Cerulean_Warbler_0014_797226.jpg', 'Cerulean_Warbler_0072_163200.jpg',
                             'Cerulean_Warbler_0080_163399.jpg'],
    '165.Chestnut_sided_Warbler': ['Chestnut_Sided_Warbler_0128_163696.jpg', 'Chestnut_Sided_Warbler_0097_163750.jpg',
                                   'Chestnut_Sided_Warbler_0094_164152.jpg', 'Chestnut_Sided_Warbler_0105_163996.jpg',
                                   'Chestnut_Sided_Warbler_0101_164324.jpg'],
    '166.Golden_winged_Warbler': ['Golden_Winged_Warbler_0079_794820.jpg', 'Golden_Winged_Warbler_0046_794828.jpg',
                                  'Golden_Winged_Warbler_0061_164516.jpg', 'Golden_Winged_Warbler_0068_794825.jpg',
                                  'Golden_Winged_Warbler_0011_794812.jpg'],
    '167.Hooded_Warbler': ['Hooded_Warbler_0040_165173.jpg', 'Hooded_Warbler_0001_164704.jpg',
                           'Hooded_Warbler_0021_165057.jpg', 'Hooded_Warbler_0058_164674.jpg',
                           'Hooded_Warbler_0053_164631.jpg'],
    '168.Kentucky_Warbler': ['Kentucky_Warbler_0008_165369.jpg', 'Kentucky_Warbler_0035_795878.jpg',
                             'Kentucky_Warbler_0050_165278.jpg', 'Kentucky_Warbler_0071_165342.jpg',
                             'Kentucky_Warbler_0072_165305.jpg'],
    '169.Magnolia_Warbler': ['Magnolia_Warbler_0041_165709.jpg', 'Magnolia_Warbler_0092_165807.jpg',
                             'Magnolia_Warbler_0029_165567.jpg', 'Magnolia_Warbler_0030_165782.jpg',
                             'Magnolia_Warbler_0053_165682.jpg'],
    '170.Mourning_Warbler': ['Mourning_Warbler_0069_166559.jpg', 'Mourning_Warbler_0035_166586.jpg',
                             'Mourning_Warbler_0002_166520.jpg', 'Mourning_Warbler_0079_166564.jpg',
                             'Mourning_Warbler_0015_166535.jpg'],
    '171.Myrtle_Warbler': ['Myrtle_Warbler_0023_166764.jpg', 'Myrtle_Warbler_0050_166820.jpg',
                           'Myrtle_Warbler_0043_166708.jpg', 'Myrtle_Warbler_0098_166794.jpg',
                           'Myrtle_Warbler_0015_166713.jpg'],
    '172.Nashville_Warbler': ['Nashville_Warbler_0108_167259.jpg', 'Nashville_Warbler_0098_167293.jpg',
                              'Nashville_Warbler_0104_167096.jpg', 'Nashville_Warbler_0110_167268.jpg',
                              'Nashville_Warbler_0081_167234.jpg'],
    '173.Orange_crowned_Warbler': ['Orange_Crowned_Warbler_0062_168119.jpg', 'Orange_Crowned_Warbler_0050_168166.jpg',
                                   'Orange_Crowned_Warbler_0055_168600.jpg', 'Orange_Crowned_Warbler_0118_167640.jpg',
                                   'Orange_Crowned_Warbler_0067_167588.jpg'],
    '174.Palm_Warbler': ['Palm_Warbler_0083_170281.jpg', 'Palm_Warbler_0012_170857.jpg', 'Palm_Warbler_0015_169626.jpg',
                         'Palm_Warbler_0126_170311.jpg', 'Palm_Warbler_0136_170276.jpg'],
    '175.Pine_Warbler': ['Pine_Warbler_0017_171678.jpg', 'Pine_Warbler_0127_171742.jpg', 'Pine_Warbler_0060_171635.jpg',
                         'Pine_Warbler_0056_172064.jpg', 'Pine_Warbler_0102_171147.jpg'],
    '176.Prairie_Warbler': ['Prairie_Warbler_0073_172771.jpg', 'Prairie_Warbler_0120_173097.jpg',
                            'Prairie_Warbler_0063_172682.jpg', 'Prairie_Warbler_0053_173290.jpg',
                            'Prairie_Warbler_0080_172724.jpg'],
    '177.Prothonotary_Warbler': ['Prothonotary_Warbler_0062_174412.jpg', 'Prothonotary_Warbler_0037_173418.jpg',
                                 'Prothonotary_Warbler_0076_174118.jpg', 'Prothonotary_Warbler_0070_174650.jpg',
                                 'Prothonotary_Warbler_0110_173857.jpg'],
    '178.Swainson_Warbler': ['Swainson_Warbler_0017_174685.jpg', 'Swainson_Warbler_0039_794859.jpg',
                             'Swainson_Warbler_0051_794900.jpg', 'Swainson_Warbler_0037_174691.jpg',
                             'Swainson_Warbler_0018_174715.jpg'],
    '179.Tennessee_Warbler': ['Tennessee_Warbler_0051_175015.jpg', 'Tennessee_Warbler_0019_174786.jpg',
                              'Tennessee_Warbler_0023_174977.jpg', 'Tennessee_Warbler_0033_174772.jpg',
                              'Tennessee_Warbler_0004_174997.jpg'],
    '180.Wilson_Warbler': ['Wilson_Warbler_0107_175320.jpg', 'Wilson_Warbler_0065_175924.jpg',
                           'Wilson_Warbler_0129_175256.jpg', 'Wilson_Warbler_0126_175368.jpg',
                           'Wilson_Warbler_0054_175285.jpg'],
    '181.Worm_eating_Warbler': ['Worm_Eating_Warbler_0063_795553.jpg', 'Worm_Eating_Warbler_0011_795566.jpg',
                                'Worm_Eating_Warbler_0092_795524.jpg', 'Worm_Eating_Warbler_0006_176037.jpg',
                                'Worm_Eating_Warbler_0018_795546.jpg'],
    '182.Yellow_Warbler': ['Yellow_Warbler_0083_176292.jpg', 'Yellow_Warbler_0096_176586.jpg',
                           'Yellow_Warbler_0119_176485.jpg', 'Yellow_Warbler_0102_176821.jpg',
                           'Yellow_Warbler_0049_176526.jpg'],
    '183.Northern_Waterthrush': ['Northern_Waterthrush_0043_177070.jpg', 'Northern_Waterthrush_0080_177080.jpg',
                                 'Northern_Waterthrush_0022_177003.jpg', 'Northern_Waterthrush_0050_177331.jpg',
                                 'Northern_Waterthrush_0014_177305.jpg'],
    '184.Louisiana_Waterthrush': ['Louisiana_Waterthrush_0087_795261.jpg', 'Louisiana_Waterthrush_0001_795271.jpg',
                                  'Louisiana_Waterthrush_0034_795242.jpg', 'Louisiana_Waterthrush_0020_795265.jpg',
                                  'Louisiana_Waterthrush_0077_795247.jpg'],
    '185.Bohemian_Waxwing': ['Bohemian_Waxwing_0046_177864.jpg', 'Bohemian_Waxwing_0042_177887.jpg',
                             'Bohemian_Waxwing_0024_177661.jpg', 'Bohemian_Waxwing_0031_796633.jpg',
                             'Bohemian_Waxwing_0048_177821.jpg'],
    '186.Cedar_Waxwing': ['Cedar_Waxwing_0094_178049.jpg', 'Cedar_Waxwing_0016_178629.jpg',
                          'Cedar_Waxwing_0125_178921.jpg', 'Cedar_Waxwing_0004_179215.jpg',
                          'Cedar_Waxwing_0065_179017.jpg'],
    '187.American_Three_toed_Woodpecker': ['American_Three_Toed_Woodpecker_0018_179831.jpg',
                                           'American_Three_Toed_Woodpecker_0007_179932.jpg',
                                           'American_Three_Toed_Woodpecker_0024_179876.jpg',
                                           'American_Three_Toed_Woodpecker_0009_179919.jpg',
                                           'American_Three_Toed_Woodpecker_0012_179905.jpg'],
    '188.Pileated_Woodpecker': ['Pileated_Woodpecker_0056_180094.jpg', 'Pileated_Woodpecker_0034_180419.jpg',
                                'Pileated_Woodpecker_0110_180521.jpg', 'Pileated_Woodpecker_0079_180388.jpg',
                                'Pileated_Woodpecker_0088_180054.jpg'],
    '189.Red_bellied_Woodpecker': ['Red_Bellied_Woodpecker_0112_180827.jpg', 'Red_Bellied_Woodpecker_0017_181131.jpg',
                                   'Red_Bellied_Woodpecker_0125_180780.jpg', 'Red_Bellied_Woodpecker_0086_181891.jpg',
                                   'Red_Bellied_Woodpecker_0020_182335.jpg'],
    '190.Red_cockaded_Woodpecker': ['Red_Cockaded_Woodpecker_0023_794701.jpg',
                                    'Red_Cockaded_Woodpecker_0033_794721.jpg',
                                    'Red_Cockaded_Woodpecker_0027_794713.jpg',
                                    'Red_Cockaded_Woodpecker_0029_794724.jpg',
                                    'Red_Cockaded_Woodpecker_0039_794736.jpg'],
    '191.Red_headed_Woodpecker': ['Red_Headed_Woodpecker_0020_183255.jpg', 'Red_Headed_Woodpecker_0005_183414.jpg',
                                  'Red_Headed_Woodpecker_0068_183662.jpg', 'Red_Headed_Woodpecker_0013_182721.jpg',
                                  'Red_Headed_Woodpecker_0095_183688.jpg'],
    '192.Downy_Woodpecker': ['Downy_Woodpecker_0040_184061.jpg', 'Downy_Woodpecker_0031_184120.jpg',
                             'Downy_Woodpecker_0090_183964.jpg', 'Downy_Woodpecker_0005_184098.jpg',
                             'Downy_Woodpecker_0136_184534.jpg'],
    '193.Bewick_Wren': ['Bewick_Wren_0083_185190.jpg', 'Bewick_Wren_0084_184715.jpg', 'Bewick_Wren_0015_184981.jpg',
                        'Bewick_Wren_0110_185216.jpg', 'Bewick_Wren_0081_185080.jpg'],
    '194.Cactus_Wren': ['Cactus_Wren_0089_186023.jpg', 'Cactus_Wren_0097_186015.jpg', 'Cactus_Wren_0025_185696.jpg',
                        'Cactus_Wren_0066_186028.jpg', 'Cactus_Wren_0033_186014.jpg'],
    '195.Carolina_Wren': ['Carolina_Wren_0113_186675.jpg', 'Carolina_Wren_0099_186237.jpg',
                          'Carolina_Wren_0014_186525.jpg', 'Carolina_Wren_0020_186702.jpg',
                          'Carolina_Wren_0128_186581.jpg'],
    '196.House_Wren': ['House_Wren_0108_187102.jpg', 'House_Wren_0107_187230.jpg', 'House_Wren_0035_187708.jpg',
                       'House_Wren_0094_187226.jpg', 'House_Wren_0122_187331.jpg'],
    '197.Marsh_Wren': ['Marsh_Wren_0056_188241.jpg', 'Marsh_Wren_0141_188796.jpg', 'Marsh_Wren_0006_188126.jpg',
                       'Marsh_Wren_0044_188270.jpg', 'Marsh_Wren_0039_188201.jpg'],
    '198.Rock_Wren': ['Rock_Wren_0122_189042.jpg', 'Rock_Wren_0063_189121.jpg', 'Rock_Wren_0069_188969.jpg',
                      'Rock_Wren_0111_189443.jpg', 'Rock_Wren_0027_189331.jpg'],
    '199.Winter_Wren': ['Winter_Wren_0066_189637.jpg', 'Winter_Wren_0030_190311.jpg', 'Winter_Wren_0075_189578.jpg',
                        'Winter_Wren_0065_189675.jpg', 'Winter_Wren_0037_190123.jpg'],
    '200.Common_Yellowthroat': ['Common_Yellowthroat_0004_190606.jpg', 'Common_Yellowthroat_0054_190398.jpg',
                                'Common_Yellowthroat_0010_190572.jpg', 'Common_Yellowthroat_0126_190407.jpg',
                                'Common_Yellowthroat_0032_190592.jpg']
}


@DATASETS.register_module()
class CUBFSCILDataset(Dataset):
    """CUB dataset for few shot class-incremental classification.
    few_cls is None when performing usual training, is tuple for few-shot training
    """

    def __init__(
            self,
            data_prefix: str,
            pipeline: List[Dict],
            num_cls: int = 200,
            subset: Literal['train', 'test'] = 'train',
            few_cls: Optional[Tuple] = None,
            test_mode: bool = False,
    ):
        self.data_prefix = data_prefix
        assert isinstance(pipeline, list), 'pipeline is type of list'
        self.pipeline = Compose(pipeline)

        self.subset = subset

        if few_cls is not None:
            assert self.subset == 'train'
            self.CLASSES = [CLASSES[_] for _ in few_cls]
            self.few_mod = True
        else:
            self.CLASSES = self.get_classes(num_cls)
            self.few_mod = False

        self.data_infos = self.load_annotations()

    # copy and paste from
    # https://github.com/icoz69/CEC-CVPR2021/blob/3a5d33/dataloader/cub200/cub200.py#L49
    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    # copy and modify from
    # https://github.com/icoz69/CEC-CVPR2021/blob/3a5d33/dataloader/cub200/cub200.py#L61
    def load_annotations(self):
        image_file = os.path.join(self.data_prefix, 'images.txt')
        split_file = os.path.join(self.data_prefix, 'train_test_split.txt')
        class_file = os.path.join(self.data_prefix, 'image_class_labels.txt')
        id2image = self.list2dict(mmcv.list_from_file(image_file))
        id2train = self.list2dict(mmcv.list_from_file(split_file))
        id2class = self.list2dict(mmcv.list_from_file(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        data_infos = []
        cls_cnt = defaultdict(lambda: 0)

        if self.subset == 'train':
            for k in train_idx:
                cls_id = int(id2class[k]) - 1
                if CLASSES[cls_id] in self.CLASSES:
                    if self.few_mod and id2image[k].split('/')[-1] not in FSCIL_SAMPLES[CLASSES[cls_id]]:
                        continue
                    info = {
                        'img_prefix': osp.join(self.data_prefix, 'images'),
                        'cls_id': cls_id, 'img_id': cls_cnt[cls_id],
                        'img_info': {'filename': id2image[k]},
                        'gt_label': np.array(cls_id, dtype=np.int64),
                    }
                    cls_cnt[cls_id] += 1
                    data_infos.append(info)

        elif self.subset == 'test':
            for k in test_idx:
                cls_id = int(id2class[k]) - 1
                if CLASSES[cls_id] in self.CLASSES:
                    info = {
                        'img_prefix': osp.join(self.data_prefix, 'images'),
                        'cls_id': cls_id, 'img_id': cls_cnt[cls_id],
                        'img_info': {'filename': id2image[k]},
                        'gt_label': np.array(cls_id, dtype=np.int64),
                    }
                    cls_cnt[cls_id] += 1
                    data_infos.append(info)
        else:
            raise NotImplementedError

        return data_infos

    @staticmethod
    def get_classes(num_cls):
        return CLASSES[:num_cls]

    @property
    def class_to_idx(self) -> Mapping:
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(CLASSES)}

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict:
        return self.prepare_data(idx)

    def prepare_data(self, idx: int) -> Dict:
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

