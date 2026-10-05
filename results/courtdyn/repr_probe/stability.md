# Probe verdict stability over 10 fold seeds (POST HOC)

## base

- H2_model: {'mixed': 3, 'UNTESTABLE': 7}
- feature: {'img_L20': 3, 'img_L16': 6, 'img_L12': 1}
- path/H1: {'True': 10}
- path/H2: {'floor-plane': 3, 'UNTESTABLE': 7}
- path/H3: {'False': 10}
- speed/H1: {'True': 10}
- speed/H2: {'UNTESTABLE': 10}
- speed/H3: {'False': 10}
- H2_delta_range: {'path/H2_delta': [-1.5519973148164738, 0.8930642119895482], 'speed/H2_delta': [-1.6044395671359708, -0.4194598854340668]}

## native

- H2_model: {'None': 10}
- feature: {'last_L32': 4, 'last_L24': 2, 'last_L20': 2, 'img_L16': 1, 'last_L28': 1}
- path/H1: {'True': 10}
- path/H2: {'None': 10}
- path/H3: {'True': 10}
- speed/H1: {'True': 10}
- speed/H2: {'None': 10}
- speed/H3: {'True': 9, 'False': 1}
- H2_delta_range: {}

## v3

- H2_model: {'mixed': 8, 'UNTESTABLE': 2}
- feature: {'last_L20': 2, 'img_L16': 2, 'last_L24': 1, 'last_L16': 4, 'img_L20': 1}
- path/H1: {'True': 10}
- path/H2: {'image-plane': 7, 'UNTESTABLE': 2, 'floor-plane': 1}
- path/H3: {'True': 10}
- speed/H1: {'True': 10}
- speed/H2: {'UNTESTABLE': 10}
- speed/H3: {'True': 10}
- H2_delta_range: {'path/H2_delta': [-0.7509755277746126, -0.10258220278367458], 'speed/H2_delta': [-0.5578319139512102, -0.14813377549725595]}

## mixunit

- H2_model: {'mixed': 3, 'UNTESTABLE': 7}
- feature: {'img_L24': 3, 'img_L20': 1, 'img_L16': 6}
- path/H1: {'True': 10}
- path/H2: {'UNTESTABLE': 8, 'floor-plane': 2}
- path/H3: {'False': 10}
- speed/H1: {'True': 10}
- speed/H2: {'image-plane': 2, 'UNTESTABLE': 8}
- speed/H3: {'False': 10}
- H2_delta_range: {'path/H2_delta': [-0.4063354792645041, 0.4345260985690448], 'speed/H2_delta': [-0.5766368216500685, 0.16106848363822346]}

H4 (v3 and mixunit same H2 verdict in both families): 1/10 seeds
