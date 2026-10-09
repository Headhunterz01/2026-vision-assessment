#include "armor_ekf/motion_tracker.hpp"
#include <iostream>
#include <stdexcept>
#include <set>
using namespace armor_ekf;
static void check(bool ok,const char * message) { if (!ok) throw std::runtime_error(message); }
static ImageObservation obs(float x,float y=100,double confidence=0.9) {
  return {{x,y},{50,25},2,confidence};
}
int main() {
  try {
    MotionTracker crossing;
    auto result = crossing.update({obs(100),obs(310,100,0.8)},0);
    const auto a = result.detection_ids[0], b = result.detection_ids[1];
    check(!result.target_switched,"First acquisition was incorrectly reported as a switch");
    for (int i = 1; i <= 18; ++i) {
      std::vector<ImageObservation> d{obs(100+12*i),obs(310-12*i,100,0.8)};
      if (i%2) std::swap(d[0],d[1]);
      result = crossing.update(d,i*0.02);
      check(result.detection_ids[i%2 ? 1 : 0]==a,"Crossing swapped target identity");
      check(result.detection_ids[i%2 ? 0 : 1]==b,"Crossing swapped neighbor identity");
      check(result.target_id==a,"Crossing changed primary");
    }
    MotionTracker gap(0.3,0.8,0.3);
    auto start = gap.update({obs(100),obs(350,200,0.8)},0);
    const auto target = start.target_id;
    for (int i = 1; i <= 8; ++i) gap.update({obs(100+10*i),obs(350,200,0.8)},i*0.02);
    for (int i = 9; i <= 14; ++i) {
      result = gap.update({obs(350,200,0.8)},i*0.02);
      check(result.selected==-1 && result.target_id==target,"Short occlusion switched to neighbor");
    }
    result = gap.update({obs(250),obs(350,200,0.8)},0.30);
    check(result.selected==0 && result.target_id==target,"Motion prediction failed to reacquire after occlusion");
    for (int i = 16; i <= 32; ++i) result = gap.update({obs(350,200,0.8)},i*0.02);
    check(result.target_id!=target && result.selected==0,"Expired target was not replaced");
    result = gap.update({obs(100)},0.1);
    check(result.target_changed,"Timestamp rewind did not reset identity");
    MotionTracker unique;
    unique.update({obs(100),obs(115,100,0.8)},0);
    result = unique.update({obs(108)},0.02);
    int matched = 0;
    for (const auto & track : unique.tracks()) matched += track.detection_index>=0;
    check(matched==1,"Multiple tracks consumed a single detection");
    MotionTracker switching;
    auto initial = switching.update({obs(100),obs(300,100,0.8)},0);
    for (int i = 1; i <= 6; ++i) result = switching.update({obs(300,100,0.8)},i*0.02);
    check(result.target_id!=initial.target_id && result.selected==0,"Confirmed visible neighbor was not selected after hold timeout");
    std::cout << "PASS: crossing identities, short occlusion, prediction reacquisition, target expiry, time rewind, one-to-one matching\n";
    return 0;
  } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
