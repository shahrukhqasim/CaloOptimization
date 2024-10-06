//
// Created by Shah Rukh Qasim on 10.07.2024.
//

#include "BoxyDetectorConstruction.hh"
#include "DetectorConstruction.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4Sphere.hh"
#include "G4UserLimits.hh"
#include "G4UniformMagField.hh"
#include "G4ThreeVector.hh"
#include "G4ThreeVector.hh"
#include "G4TransportationManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4Sphere.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4UniformMagField.hh"
#include "G4UserLimits.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VisAttributes.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4ChordFinder.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4PropagatorInField.hh"
#include "G4ClassicalRK4.hh"


#include <iostream>

G4VPhysicalVolume *BoxyDetectorConstruction::Construct() {

    G4NistManager* nist = G4NistManager::Instance();
    auto Mn  = new G4Element("Manganese", "Mn", 25,     54.94*g/mole);
    auto Cr  = new G4Element("Chromium",  "Cr", 24,     52.00*g/mole);
    auto Ni  = new G4Element("Nickel",    "Ni", 28,     58.70*g/mole);

    auto stainlessSteelM = new G4Material("StainlessSteel",8.02*g/cm3, 5);
    stainlessSteelM->AddElement(Mn, 0.02);
    stainlessSteelM->AddMaterial(nist->FindOrBuildMaterial("G4_Si"), 0.01);
    stainlessSteelM->AddElement(Cr, 0.19);
    stainlessSteelM->AddElement(Ni, 0.10);
    stainlessSteelM->AddMaterial(nist->FindOrBuildMaterial("G4_Fe"), 0.68);


    double limit_world_time_max_ = 5000 * ns;
    double limit_world_energy_max_ = 100 * eV;

    // Create a user limits object with a maximum step size of 1 mm
    auto userLimits = getLimitsFromDetectorConfig(detectorData);

    // Get NIST material manager

    // Define the world material
    G4Material* worldMaterial = nist->FindOrBuildMaterial("G4_AIR");
    // Get the world size from the JSON variable
    G4double worldSizeX = detectorData["worldSizeX"].asDouble() * m;
    G4double worldSizeY = detectorData["worldSizeY"].asDouble() * m;
    G4double worldSizeZ = detectorData["worldSizeZ"].asDouble() * m;

    G4double worldPositionX = detectorData["worldPositionX"].asDouble() * m;
    G4double worldPositionY = detectorData["worldPositionY"].asDouble() * m;
    G4double worldPositionZ = detectorData["worldPositionZ"].asDouble() * m;

    // Create the world volume
    G4Box* solidWorld = new G4Box("WorldX", worldSizeX / 2, worldSizeY / 2, worldSizeZ / 2);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, worldMaterial, "WorldY");
    logicWorld->SetUserLimits(userLimits);

    G4VPhysicalVolume* physWorld = new G4PVPlacement(0, G4ThreeVector(worldPositionX, worldPositionY, worldPositionZ), logicWorld, "WorldZ", 0, false, 0, true);
    slimFilmSensitiveDetector = new SlimFilmSensitiveDetector("the_sensitive_guy", detectorData["store_full"].asBool());

    std::cout<<"Nuclear interaction length of " << "G4_W"<< ": "<< nist->FindOrBuildMaterial("G4_W")->GetNuclearInterLength() / m <<std::endl;
    std::cout<<"Radiation length of " << "G4_W"<< ": "<< nist->FindOrBuildMaterial("G4_W")->GetRadlen() / m <<std::endl;


    // Process the components from the JSON variable
    const Json::Value components = detectorData["layers"];
    for (const auto& component : components) {

        std::cout<<"Adding box"<<std::endl;
        // Get the material for the component
        std::string materialName = component["material"].asString();
        G4Material* boxMaterial;
        if (materialName == "stainless_steel")
            boxMaterial = stainlessSteelM;
        else
            boxMaterial = nist->FindOrBuildMaterial(materialName);

        std::cout<<"Nuclear interaction length of " << materialName<< ": "<< boxMaterial->GetNuclearInterLength() / m <<std::endl;
        std::cout<<"Radiation length of " << materialName<< ": "<< boxMaterial->GetRadlen() / m <<std::endl;

        // Get the dimensions of the box
        G4double boxSizeX = component["dx"].asDouble() * m;
        G4double boxSizeY = component["dy"].asDouble() * m;
        G4double boxSizeZ = component["dz"].asDouble() * m;

        // Get the position of the box
        G4double posX = 0;
        G4double posY = 0;
        G4double posZ = component["z_center"].asDouble() * m;

        // Create the box volume
        G4Box* solidBox = new G4Box("BoxX", boxSizeX / 2, boxSizeY / 2, boxSizeZ / 2);
        G4LogicalVolume* logicBox = new G4LogicalVolume(solidBox, boxMaterial, "BoxY");
        auto userLimits2 = getLimitsFromDetectorConfig(component);

        // Associate the user limits with the logical volume
        logicBox->SetUserLimits(userLimits2);
        std::cout<<"Setting "<<component["limits"]["max_step_length"].asDouble() * m<<std::endl;


        G4VPhysicalVolume* physVol  = new G4PVPlacement(0, G4ThreeVector(posX, posY, posZ), logicBox, "BoxZ", logicWorld, false, 0, true);

//        if (materialName == "G4_Si") {
        logicBox->SetSensitiveDetector(slimFilmSensitiveDetector);
        int layerNumber = component["layer_number"].asInt();
        slimFilmSensitiveDetector->addSensitiveLayerInfo(physVol->GetInstanceID(), layerNumber);
//        }
    }

    // Return the physical world
    return physWorld;
}



BoxyDetectorConstruction::BoxyDetectorConstruction(Json::Value detector_data) {
    detectorData = detector_data;
}

void BoxyDetectorConstruction::setMagneticFieldValue(double strength, double theta, double phi) {
//    DetectorConstruction::setMagneticFieldValue(strength, theta, phi);
std::cout<<"cannot set magnetic field value for boxy detector.\n"<<std::endl;
}

void BoxyDetectorConstruction::ConstructSDandField() {
    G4VUserDetectorConstruction::ConstructSDandField();
    auto* sdManager = G4SDManager::GetSDMpointer();
    sdManager->AddNewDetector(slimFilmSensitiveDetector);
    std::cout<<"Sensitive set...\n";
}
