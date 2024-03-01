from jakteristics import las_utils, compute_features, FEATURE_NAMES

input_path = "../ai-classificator/spezia-section-02-secondotest.las"
xyz = las_utils.read_las_xyz(input_path)

#features = compute_features(xyz, search_radius=0.15)

output_path = "../ai-classificator/spezia-section-02-secondotest-finale-raggiogrosso.las"
#las_utils.write_with_extra_dims(input_path, output_path, features, FEATURE_NAMES)

features = compute_features(xyz, search_radius = 1.5, num_threads = 4, feature_names=['omnivariance','linearity','planarity','sphericity'])

las_utils.write_with_extra_dims(input_path, output_path, features,['omnivariance','linearity','planarity','sphericity'])

# or for a specific feature:
#omnivariance = compute_features(xyz, search_radius=0.15, feature_names=["omnivariance"])
#output_omnivariance = "/path/to/output_omnivariance.las"
#las_utils.write_with_extra_dims(input_path, output_omnivariance, omnivariance, ["omnivariance"])
