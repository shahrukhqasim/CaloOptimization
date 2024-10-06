#include <pybind11/pybind11.h>
#include "G4PhysicsListHelper.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4UserSpecialCuts.hh"
#include "G4StepLimiter.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4RunManager.hh"
#include "CustomSteppingAction.hh"
#include "DetectorConstruction.hh"
#include "G4UImanager.hh"
#include "PrimaryGeneratorAction.cc"
#include "FTFP_BERT.hh"
#include "CustomEventAction.hh"
#include "BoxyDetectorConstruction.hh"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <G4UIExecutive.hh>
#include "QGSP_BERT.hh"
#include "json/json.h"
#include <iostream>
#include <sstream>
#include <stdexcept> // For standard exceptions like std::runtime_error


namespace py = pybind11;
using namespace py::literals;


G4RunManager* runManager;
G4UImanager *ui_manager;
PrimaryGeneratorAction *primariesGenerator;
DetectorConstruction * detector;
CustomSteppingAction * steppingAction;
//bool collect_full_data;
CLHEP::MTwistEngine *randomEngine;
CustomEventAction *customEventAction;



int add(int a, int b) {
    return a + b;
}

void simulate_particle(double px, double py, double pz, int pdgid,
                    double x, double y, double z) {
    if (ui_manager == nullptr) {
        G4cout<<"Call initialize(...) before running this function.\n";
        throw std::runtime_error("Forgot to call initialize?");
    }

    primariesGenerator->setNextMomenta(px, py, pz);
    primariesGenerator->setNextPosition(x, y, z);
    primariesGenerator->setNextPdgId(pdgid);

    ui_manager->ApplyCommand(std::string("/run/beamOn ") + std::to_string(1));


}

void set_kill_momenta(double kill_momenta) {
    steppingAction->setKillMomenta(kill_momenta);
}

py::dict collect_deposits() {
    auto detector2 = dynamic_cast<BoxyDetectorConstruction*>(detector);
    if (detector2 == nullptr) {
        throw std::runtime_error("Sensitive film only possible for GDetectorConstruction.");
    }

    if (detector2->slimFilmSensitiveDetector == nullptr) {
        throw std::runtime_error("Slim film not installed in the detector.");
    }

    std::vector<double>& chargeDeposit = detector2->slimFilmSensitiveDetector->chargeDeposit;
//    std::vector<int>& layer = detector2->slimFilmSensitiveDetector->layer;

    std::vector<double> chargeDeposit_copy(chargeDeposit.begin(), chargeDeposit.end());
//    std::vector<int> layer_copy(layer.begin(), layer.end());

    py::array np_chargeDeposit = py::cast(chargeDeposit_copy);
//    py::array np_layer = py::cast(layer_copy);

//    py::array np_layer = py::array_t<int>(layer_copy);


    py::dict d = py::dict(
            "charge_deposit"_a = np_chargeDeposit
//            "layer"_a = np_layer
    );

    return d;
}

py::dict collect_full_deposits() {
    auto detector2 = dynamic_cast<BoxyDetectorConstruction*>(detector);
    if (detector2 == nullptr) {
        throw std::runtime_error("Sensitive film only possible for GDetectorConstruction.");
    }

    if (detector2->slimFilmSensitiveDetector == nullptr) {
        throw std::runtime_error("Slim film not installed in the detector.");
    }

    std::vector<double>& depositFullX = detector2->slimFilmSensitiveDetector->depositFullX;
    std::vector<double>& depositFullY = detector2->slimFilmSensitiveDetector->depositFullY;
    std::vector<double>& depositFullZ = detector2->slimFilmSensitiveDetector->depositFullZ;
    std::vector<double>& depositFullCharge = detector2->slimFilmSensitiveDetector->depositFullCharge;

    // Create copies of the vectors to convert to numpy arrays
    std::vector<double> depositFullX_copy(depositFullX.begin(), depositFullX.end());
    std::vector<double> depositFullY_copy(depositFullY.begin(), depositFullY.end());
    std::vector<double> depositFullZ_copy(depositFullZ.begin(), depositFullZ.end());
    std::vector<double> depositFullCharge_copy(depositFullCharge.begin(), depositFullCharge.end());

    // Convert vectors to numpy arrays
    py::array np_depositFullX = py::cast(depositFullX_copy);
    py::array np_depositFullY = py::cast(depositFullY_copy);
    py::array np_depositFullZ = py::cast(depositFullZ_copy);
    py::array np_depositFullCharge = py::cast(depositFullCharge_copy);

    // Create a dictionary with the numpy arrays
    py::dict d = py::dict(
            "deposit_full_x"_a = np_depositFullX,
            "deposit_full_y"_a = np_depositFullY,
            "deposit_full_z"_a = np_depositFullZ,
            "deposit_full_charge"_a = np_depositFullCharge
    );

    return d;
}


std::string initialize( int rseed_0,
                 int rseed_1, int rseed_2, int rseed_3, std::string detector_specs) {
    randomEngine = new CLHEP::MTwistEngine(rseed_0);


    long seeds[4] = {rseed_0, rseed_1, rseed_2, rseed_3};

    CLHEP::HepRandom::setTheSeeds(seeds);
    G4Random::setTheSeeds(seeds);

    runManager = new G4RunManager;


    bool applyStepLimiter = true;
    bool storeAll = false;
    bool storePrimary = true;
    if (detector_specs.empty())
        detector = new DetectorConstruction();
    else {
        std::cout<<"Exa check \n";
        Json::Value detectorData;
        Json::CharReaderBuilder readerBuilder;
        std::string errs;

        std::istringstream iss(detector_specs);
        if (Json::parseFromStream(readerBuilder, iss, &detectorData, &errs)) {
            // Output the parsed JSON object
            std::cout << detectorData["worldSizeX"] << std::endl;
        } else {
            std::cerr << "Failed to parse JSON: " << errs << std::endl;
        }


        int type = detectorData["type"].asInt();
//        applyStepLimiter = (detectorData["limits"]["max_step_length"].asDouble() > 0);
        if (type == 0)
            detector = new BoxyDetectorConstruction(detectorData);
        else
            throw std::runtime_error("Invalid detector type specified.");

        if (detectorData.isMember("store_all")) {
            storeAll = detectorData["store_all"].asBool();
        }
        if (detectorData.isMember("store_primary")) {
            storePrimary = detectorData["store_primary"].asBool();
        }
    }

    std::cout<<"Detector initializing..."<<std::endl;
    runManager->SetUserInitialization(detector);

    auto physicsList = new FTFP_BERT;
//    auto physicsList = new QGSP_BERT_HP_PEN();
//    auto physicsList = new QGSP_BERT;
    std::cout<<"Step limiter physics applied: "<<applyStepLimiter<<std::endl;
    if (applyStepLimiter) {
        physicsList->RegisterPhysics(new G4StepLimiterPhysics());
    }
    runManager->SetUserInitialization(physicsList);

    customEventAction = new CustomEventAction();
    primariesGenerator = new PrimaryGeneratorAction();
    steppingAction = new CustomSteppingAction();
    primariesGenerator->setSteppingAction(steppingAction);
    customEventAction->setSteppingAction(steppingAction);
    steppingAction->setStoreAll(storeAll);
    steppingAction->setStorePrimary(storePrimary);

//    auto actionInitialization = new B4aActionInitialization(detector, eventAction, primariesGenerator);
//    runManager->SetUserInitialization(actionInitialization);

    runManager->SetUserAction(primariesGenerator);
    runManager->SetUserAction(steppingAction);
    runManager->SetUserAction(customEventAction);

    // Get the pointer to the User Interface manager
    ui_manager = G4UImanager::GetUIpointer();




    ui_manager->ApplyCommand(std::string("/run/initialize"));
    ui_manager->ApplyCommand(std::string("/run/printProgress 100"));

    std::cout<<"Initialized"<<std::endl;

    Json::Value returnData;
    returnData["weight_total"] = detector->getDetectorWeight();

    Json::StreamWriterBuilder writer;
    writer["indentation"] = ""; // No indentation (compact representation)

    // Convert JSON value to string
    std::string output = Json::writeString(writer, returnData);

    return output;
}

void kill_secondary_tracks(bool do_kill) {
    steppingAction->setKillSecondary(do_kill);
}

void visualize() {
    // Interactive mode
    ui_manager->ApplyCommand("/vis/open OGLIX 600x600-0+0");
    ui_manager->ApplyCommand("/vis/viewer/set/autoRefresh true");
    ui_manager->ApplyCommand("/vis/scene/add/axes 0 0 0 10 cm");
    ui_manager->ApplyCommand("/vis/viewer/set/style wireframe");
    ui_manager->ApplyCommand("/vis/viewer/set/hiddenMarker true");
    ui_manager->ApplyCommand("/vis/viewer/set/viewpointThetaPhi 60 30");
    ui_manager->ApplyCommand("/vis/drawVolume");
    ui_manager->ApplyCommand("/vis/viewer/zoom 0.7");
    ui_manager->ApplyCommand("/vis/viewer/update");
    ui_manager->ApplyCommand("/run/initialize");
//    auto ui = new G4UIExecutive(1, nullptr);
//    ui->SessionStart();
//    ui->SessionStart();
//    G4UIExecutive* ui = nullptr;
//    if (argc == 1) {
//        ui = new G4UIExecutive(argc, argv);
//    }

}

PYBIND11_MODULE(sampling_calo, m) {
    m.def("add", &add, "A function which adds two numbers");
    m.def("simulate_particle", &simulate_particle, "A function which simulates a particle through the detector");
    m.def("initialize", &initialize, "Initialize geant4 stuff");
    m.def("set_kill_momenta", &set_kill_momenta, "Set the kill momenta");
    m.def("kill_secondary_tracks", &kill_secondary_tracks, "Kill all tracks from resulting cascade");
    m.def("visualize", &visualize, "Visualize");
    m.def("collect_deposits", &collect_deposits, "Collect charge deposits");
    m.def("collect_full_deposits", &collect_full_deposits, "Collect full charge deposits with positions");

}

// Compile the C++ code to a shared library
// c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` my_functions.cpp -o my_functions`python3-config --extension-suffix`
