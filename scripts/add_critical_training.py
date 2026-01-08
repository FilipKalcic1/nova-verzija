"""
Add critical training examples for primary tools.
These examples will improve LLM routing accuracy.
"""

import json
from pathlib import Path

# Critical examples for PRIMARY_TOOLS that are missing or underrepresented
CRITICAL_EXAMPLES = [
    # === get_MasterData - Vehicle Info ===
    {"query": "kolika mi je kilometraža", "intent": "GET_MILEAGE", "primary_tool": "get_MasterData", "extract_fields": ["LastMileage"], "category": "vehicle_info"},
    {"query": "koliko imam kilometara", "intent": "GET_MILEAGE", "primary_tool": "get_MasterData", "extract_fields": ["LastMileage"], "category": "vehicle_info"},
    {"query": "koja mi je kilometraža", "intent": "GET_MILEAGE", "primary_tool": "get_MasterData", "extract_fields": ["LastMileage"], "category": "vehicle_info"},
    {"query": "trenutna kilometraža", "intent": "GET_MILEAGE", "primary_tool": "get_MasterData", "extract_fields": ["LastMileage"], "category": "vehicle_info"},
    {"query": "moja kilometraža", "intent": "GET_MILEAGE", "primary_tool": "get_MasterData", "extract_fields": ["LastMileage"], "category": "vehicle_info"},
    {"query": "stanje kilometara", "intent": "GET_MILEAGE", "primary_tool": "get_MasterData", "extract_fields": ["LastMileage"], "category": "vehicle_info"},
    {"query": "podaci o mom vozilu", "intent": "GET_VEHICLE_INFO", "primary_tool": "get_MasterData", "extract_fields": ["FullVehicleName", "LicencePlate", "LastMileage"], "category": "vehicle_info"},
    {"query": "informacije o vozilu", "intent": "GET_VEHICLE_INFO", "primary_tool": "get_MasterData", "extract_fields": ["FullVehicleName", "LicencePlate"], "category": "vehicle_info"},
    {"query": "moje vozilo", "intent": "GET_VEHICLE_INFO", "primary_tool": "get_MasterData", "extract_fields": ["FullVehicleName", "LicencePlate"], "category": "vehicle_info"},
    {"query": "koje vozilo imam", "intent": "GET_VEHICLE_INFO", "primary_tool": "get_MasterData", "extract_fields": ["FullVehicleName"], "category": "vehicle_info"},
    {"query": "koja su mi tablica", "intent": "GET_PLATE", "primary_tool": "get_MasterData", "extract_fields": ["LicencePlate"], "category": "vehicle_info"},
    {"query": "registarske oznake", "intent": "GET_PLATE", "primary_tool": "get_MasterData", "extract_fields": ["LicencePlate"], "category": "vehicle_info"},
    {"query": "kada mi istječe registracija", "intent": "GET_REGISTRATION", "primary_tool": "get_MasterData", "extract_fields": ["RegistrationExpirationDate"], "category": "vehicle_info"},
    {"query": "do kad vrijedi registracija", "intent": "GET_REGISTRATION", "primary_tool": "get_MasterData", "extract_fields": ["RegistrationExpirationDate"], "category": "vehicle_info"},
    {"query": "koja je moja lizing kuća", "intent": "GET_LEASING", "primary_tool": "get_MasterData", "extract_fields": ["ProviderName"], "category": "vehicle_info"},
    {"query": "tko mi je leasing provider", "intent": "GET_LEASING", "primary_tool": "get_MasterData", "extract_fields": ["ProviderName"], "category": "vehicle_info"},
    
    # === post_AddMileage - Mileage Input ===
    {"query": "unesi kilometražu", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    {"query": "upiši kilometražu", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    {"query": "unesi km", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    {"query": "upiši km", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    {"query": "unesi 15000 km", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    {"query": "nova kilometraža 25000", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    {"query": "želim unijeti kilometražu", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    {"query": "moram upisati km", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    {"query": "hoću unijeti km", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    {"query": "ažuriraj kilometražu", "intent": "INPUT_MILEAGE", "primary_tool": "post_AddMileage", "extract_fields": [], "category": "mileage_input"},
    
    # === get_VehicleCalendar - My Bookings ===
    {"query": "moje rezervacije", "intent": "GET_MY_BOOKINGS", "primary_tool": "get_VehicleCalendar", "extract_fields": ["FromTime", "ToTime", "VehicleName"], "category": "booking"},
    {"query": "koje rezervacije imam", "intent": "GET_MY_BOOKINGS", "primary_tool": "get_VehicleCalendar", "extract_fields": ["FromTime", "ToTime"], "category": "booking"},
    {"query": "pokaži moje rezervacije", "intent": "GET_MY_BOOKINGS", "primary_tool": "get_VehicleCalendar", "extract_fields": ["FromTime", "ToTime"], "category": "booking"},
    {"query": "imam li kakvu rezervaciju", "intent": "GET_MY_BOOKINGS", "primary_tool": "get_VehicleCalendar", "extract_fields": ["FromTime", "ToTime"], "category": "booking"},
    {"query": "kada imam rezervirano vozilo", "intent": "GET_MY_BOOKINGS", "primary_tool": "get_VehicleCalendar", "extract_fields": ["FromTime", "ToTime"], "category": "booking"},
    {"query": "prikazi sve rezervacije", "intent": "GET_MY_BOOKINGS", "primary_tool": "get_VehicleCalendar", "extract_fields": ["FromTime", "ToTime"], "category": "booking"},
    
    # === get_AvailableVehicles + post_VehicleCalendar - Booking Flow ===
    {"query": "trebam rezervirati vozilo", "intent": "BOOK_VEHICLE", "primary_tool": "get_AvailableVehicles", "extract_fields": [], "category": "booking"},
    {"query": "trebam auto za sutra", "intent": "BOOK_VEHICLE", "primary_tool": "get_AvailableVehicles", "extract_fields": [], "category": "booking"},
    {"query": "želim rezervirati auto", "intent": "BOOK_VEHICLE", "primary_tool": "get_AvailableVehicles", "extract_fields": [], "category": "booking"},
    {"query": "rezerviraj mi vozilo", "intent": "BOOK_VEHICLE", "primary_tool": "get_AvailableVehicles", "extract_fields": [], "category": "booking"},
    {"query": "slobodna vozila za sutra", "intent": "CHECK_AVAILABILITY", "primary_tool": "get_AvailableVehicles", "extract_fields": [], "category": "booking"},
    {"query": "dostupna vozila", "intent": "CHECK_AVAILABILITY", "primary_tool": "get_AvailableVehicles", "extract_fields": [], "category": "booking"},
    {"query": "koja vozila su slobodna", "intent": "CHECK_AVAILABILITY", "primary_tool": "get_AvailableVehicles", "extract_fields": [], "category": "booking"},
    {"query": "hoću rezervirati za petak", "intent": "BOOK_VEHICLE", "primary_tool": "get_AvailableVehicles", "extract_fields": [], "category": "booking"},
    
    # === post_AddCase - Damage/Issue Reporting ===
    {"query": "prijavi štetu", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "imam štetu na vozilu", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "prijavi kvar", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "imam kvar", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "nešto ne radi na autu", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "dogodila se nesreća", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "imao sam sudar", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "ogrebao sam auto", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "udario sam vozilo", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "problem s motorom", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "auto se pokvario", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "želim prijaviti štetu", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    {"query": "trebam prijaviti kvar", "intent": "REPORT_DAMAGE", "primary_tool": "post_AddCase", "extract_fields": [], "category": "case"},
    
    # === delete_VehicleCalendar_id - Cancel Booking ===
    {"query": "otkaži rezervaciju", "intent": "CANCEL_BOOKING", "primary_tool": "delete_VehicleCalendar_id", "extract_fields": [], "category": "booking"},
    {"query": "obriši rezervaciju", "intent": "CANCEL_BOOKING", "primary_tool": "delete_VehicleCalendar_id", "extract_fields": [], "category": "booking"},
    {"query": "poništi rezervaciju", "intent": "CANCEL_BOOKING", "primary_tool": "delete_VehicleCalendar_id", "extract_fields": [], "category": "booking"},
    {"query": "želim otkazati rezervaciju", "intent": "CANCEL_BOOKING", "primary_tool": "delete_VehicleCalendar_id", "extract_fields": [], "category": "booking"},
]


def main():
    """Add critical examples to training_queries.json."""
    training_path = Path(__file__).parent.parent / "data" / "training_queries.json"
    
    # Load existing
    with open(training_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    existing_queries = {ex["query"].lower() for ex in data["examples"]}
    
    # Add new examples (skip duplicates)
    added = 0
    for ex in CRITICAL_EXAMPLES:
        if ex["query"].lower() not in existing_queries:
            # Add with full structure
            full_example = {
                "query": ex["query"],
                "intent": ex["intent"],
                "primary_tool": ex["primary_tool"],
                "alternative_tools": [],
                "extract_fields": ex.get("extract_fields", []),
                "response_template": "",
                "category": ex.get("category", "general")
            }
            data["examples"].insert(0, full_example)  # Add to beginning for priority
            existing_queries.add(ex["query"].lower())
            added += 1
    
    # Save
    with open(training_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Added {added} critical training examples")
    print(f"📊 Total examples: {len(data['examples'])}")
    
    # Summary by tool
    from collections import Counter
    tool_counts = Counter(ex["primary_tool"] for ex in data["examples"])
    print("\n🔧 Primary tool distribution (top 10):")
    for tool, count in tool_counts.most_common(10):
        print(f"  {tool}: {count}")


if __name__ == "__main__":
    main()
