import numpy as np
import cv2 as cv
import argparse

def visualize(image, faces, thickness=2):
    for idx, face in enumerate(faces[1]):
        coords = face[:-1].astype(np.int32)
        cv.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
        cv.circle(image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
        cv.circle(image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
        cv.circle(image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
        cv.circle(image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
        cv.circle(image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reference_image", required=True, help="Path to input Aadhaar reference image")
ap.add_argument("-q", "--query_image", required=True, help="Path to input query image")
args = vars(ap.parse_args())

reference_image = cv.imread(args["reference_image"])
query_image = cv.imread(args["query_image"])

faceDetector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (reference_image.shape[1], reference_image.shape[0]), score_threshold=0.9, nms_threshold=0.3, top_k=5000)
faceInAadhar = faceDetector.detect(reference_image)

visualize(reference_image, faceInAadhar)

cv.imshow("faceInAadhar", reference_image)
cv.waitKey(0)
faceDetector.setInputSize((query_image.shape[1],query_image.shape[0]))
faceInQuery=faceDetector.detect(query_image)
visualize(query_image, faceInQuery)
cv.imshow("faceInquery", query_image)
cv.waitKey(0)

recognizer=cv.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx","")
face1_align=recognizer.alignCrop(reference_image,faceInAadhar[1][0])
face2_align=recognizer.alignCrop(query_image,faceInQuery[1][0])

face1_feature=recognizer.feature(face1_align)
face2_feature=recognizer.feature(face2_align)

cosine_score=recognizer.match(face1_feature,face2_feature,cv.FACE_RECOGNIZER_SF_FR_COSINE)
l2_score=recognizer.match(face1_feature,face2_feature,cv.FaceRecognizerSF_FR_NORM_L2)

cosine_similarity_threshold=0.363
l2_similarity_threshold=1.128

# msg='different identities'
# if cosine_score>=cosine_similarity_threshold:
#     msg='the same identity'
#     print('They have {}.Cosine Similarity: {},threshold: {} (higher value means higher similarity, max 1.0).'.format(msg,cosine_score,cosine_similarity_threshold))

# msg='different identities'
# if l2_score>=cosine_similarity_threshold:
#     msg='the same identity'
#     print('They have {}.NormL2 Distance: {},threshold: {} (lower value means lower similarity, min 0.0).'.format(msg,l2_score,l2_similarity_threshold))
    
msg = 'different identities'
if cosine_score >= cosine_similarity_threshold:
    msg = 'the same identity'
print('They have {}. Cosine Similarity: {}, threshold: {} (higher value means higher similarity, max 1.0).'.format(msg, cosine_score, cosine_similarity_threshold))

msg = 'different identities'
if l2_score <= l2_similarity_threshold:
    msg = 'the same identity'
print('They have {}. NormL2 Distance: {}, threshold: {} (lower value means higher similarity, min 0.0).'.format(msg, l2_score, l2_similarity_threshold))