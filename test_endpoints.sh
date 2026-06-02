#!/bin/bash
echo "1. Testing Single Prediction (/api/v1/predict)"
curl -s -X POST "http://localhost:8888/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"mxene_type": "Ti3C2Tx", "terminations": "O", "electrolyte": "H2SO4", "thickness_um": 5.0, "deposition_method": "vacuum_filtration"}'

echo -e "\n\n2. Testing Batch Prediction (/api/v1/predict/batch)"
curl -s -X POST "http://localhost:8888/api/v1/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"requests": [
          {"mxene_type": "Ti3C2Tx", "terminations": "O", "electrolyte": "H2SO4", "thickness_um": 5.0, "deposition_method": "vacuum_filtration"},
          {"mxene_type": "Mo2CTx", "terminations": "F", "electrolyte": "KOH", "thickness_um": 10.0, "deposition_method": "drop_casting"}
         ]}'

echo -e "\n\n3. Testing Models Metrics (/api/v1/models/metrics)"
curl -s -X GET "http://localhost:8888/api/v1/models/metrics"

echo -e "\n\n4. Testing GET Devices (/api/v1/devices)"
curl -s -X GET "http://localhost:8888/api/v1/devices"
