import cv2


im = cv2.imread('../screenshots/14.jpg')
mask = cv2.imread('../screenshots/mask.jpg')
cv2.imwrite('14.ppm', im)
cv2.imwrite('mask.ppm', mask)
