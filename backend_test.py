import requests
import sys
import time
import json
from datetime import datetime

class GNSSMissionControlTester:
    def __init__(self, base_url="https://satellite-tracker-3.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=10):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, list):
                        print(f"   Response: List with {len(response_data)} items")
                    elif isinstance(response_data, dict):
                        print(f"   Response keys: {list(response_data.keys())}")
                    return True, response_data
                except:
                    print(f"   Response: {response.text[:200]}...")
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout after {timeout}s")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_satellites_endpoint(self):
        """Test GET /api/satellites"""
        success, response = self.run_test(
            "Get Satellites List",
            "GET",
            "satellites",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} satellites")
            if len(response) >= 10:
                print("   ✅ Expected 10 satellites found")
                # Check constellation distribution
                constellations = {}
                for sat in response:
                    const = sat.get('constellation', 'Unknown')
                    constellations[const] = constellations.get(const, 0) + 1
                print(f"   Constellation distribution: {constellations}")
                return response
            else:
                print(f"   ⚠️  Expected 10 satellites, found {len(response)}")
        
        return response if success else []

    def test_predictions_endpoint(self):
        """Test GET /api/predictions"""
        success, response = self.run_test(
            "Get Predictions Data",
            "GET",
            "predictions",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} prediction data points")
            if len(response) > 0:
                sample = response[0]
                expected_keys = ['timestamp', 'satellite_id', 'constellation', 'pred_clock_error_m', 'pred_orbit_error_m']
                has_keys = all(key in sample for key in expected_keys)
                print(f"   ✅ Sample prediction has required keys: {has_keys}")
                if has_keys:
                    print(f"   Sample: {sample['satellite_id']} - Clock: {sample['pred_clock_error_m']:.4f}m, Orbit: {sample['pred_orbit_error_m']:.2f}m")
        
        return response if success else []

    def test_satellite_summary(self, satellite_id):
        """Test GET /api/satellite/{id}/summary"""
        success, response = self.run_test(
            f"Get Satellite Summary for {satellite_id}",
            "GET",
            f"satellite/{satellite_id}/summary",
            200
        )
        
        if success and isinstance(response, dict):
            if 'satellite' in response and 'summary' in response:
                summary = response['summary']
                print(f"   Peak Clock Error: {summary.get('peak_clock_error', 0):.4f}m")
                print(f"   Peak Orbit Error: {summary.get('peak_orbit_error', 0):.2f}m")
                print(f"   Data Points: {summary.get('data_points', 0)}")
                return True
        
        return False

    def test_ai_agent_query(self):
        """Test POST /api/agent/query and polling"""
        query_data = {
            "prompt": "Compare G01 vs R01 clock error volatility",
            "context": {"satellite_ids": ["G01", "R01"]}
        }
        
        success, response = self.run_test(
            "Submit AI Agent Query",
            "POST",
            "agent/query",
            200,
            data=query_data,
            timeout=15
        )
        
        if success and isinstance(response, dict) and 'job_id' in response:
            job_id = response['job_id']
            print(f"   Job ID: {job_id}")
            
            # Poll for results
            print("   Polling for AI Agent results...")
            max_polls = 10
            poll_count = 0
            
            while poll_count < max_polls:
                time.sleep(3)  # Wait 3 seconds between polls
                poll_count += 1
                
                result_success, result_response = self.run_test(
                    f"Poll AI Agent Result (attempt {poll_count})",
                    "GET",
                    f"agent/result/{job_id}",
                    200,
                    timeout=10
                )
                
                if result_success and isinstance(result_response, dict):
                    status = result_response.get('status', 'unknown')
                    print(f"   Job Status: {status}")
                    
                    if status == 'completed':
                        results = result_response.get('results', {})
                        if results:
                            print("   ✅ AI Agent completed successfully")
                            if 'text_summary' in results:
                                print(f"   Summary: {results['text_summary'][:100]}...")
                            if 'statistics' in results:
                                print(f"   Statistics: {results['statistics']}")
                            return True
                        break
                    elif status == 'failed':
                        error = result_response.get('error', 'Unknown error')
                        print(f"   ❌ AI Agent failed: {error}")
                        break
                    elif status in ['pending', 'running']:
                        print(f"   ⏳ Still {status}...")
                        continue
                else:
                    print(f"   ❌ Failed to get job status")
                    break
            
            if poll_count >= max_polls:
                print(f"   ⚠️  AI Agent did not complete within {max_polls * 3} seconds")
        
        return False

    def test_websocket_endpoint(self):
        """Test WebSocket connection (basic connectivity)"""
        try:
            import websocket
            ws_url = self.base_url.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws'
            print(f"\n🔍 Testing WebSocket Connection...")
            print(f"   URL: {ws_url}")
            
            # Simple connection test
            ws = websocket.create_connection(ws_url, timeout=5)
            print("   ✅ WebSocket connection established")
            ws.close()
            self.tests_passed += 1
            self.tests_run += 1
            return True
        except Exception as e:
            print(f"   ❌ WebSocket connection failed: {str(e)}")
            self.tests_run += 1
            return False

def main():
    print("🚀 Starting GNSS Mission Control Backend Tests")
    print("=" * 60)
    
    tester = GNSSMissionControlTester()
    
    # Test 1: Get satellites
    satellites = tester.test_satellites_endpoint()
    
    # Test 2: Get predictions
    predictions = tester.test_predictions_endpoint()
    
    # Test 3: Test satellite summary for first few satellites
    if satellites:
        for i, satellite in enumerate(satellites[:3]):  # Test first 3 satellites
            satellite_id = satellite.get('satellite_id')
            if satellite_id:
                tester.test_satellite_summary(satellite_id)
    
    # Test 4: Test AI Agent functionality
    tester.test_ai_agent_query()
    
    # Test 5: Test WebSocket (if websocket-client is available)
    try:
        tester.test_websocket_endpoint()
    except ImportError:
        print("\n🔍 Testing WebSocket Connection...")
        print("   ⚠️  websocket-client not available, skipping WebSocket test")
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 Backend Tests Summary:")
    print(f"   Tests Run: {tester.tests_run}")
    print(f"   Tests Passed: {tester.tests_passed}")
    print(f"   Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All backend tests passed!")
        return 0
    else:
        print("⚠️  Some backend tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())