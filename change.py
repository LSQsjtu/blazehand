xmax = max(
            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
        )
        xmin = min(
            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
        )
        ymax = max(
            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
        )
        ymin = min(
            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
        )