"""
Tests for ConfirmationDialog - parameter formatting, modification parsing,
and confirmation message generation.
"""

import pytest
from datetime import datetime
from services.confirmation_dialog import ConfirmationDialog, ParameterDisplay


@pytest.fixture
def dialog():
    return ConfirmationDialog()


class TestFormatDatetime:
    def test_iso_string(self, dialog):
        result = dialog._format_datetime("2025-06-15T10:30:00Z")
        assert "15" in result
        assert "2025" in result

    def test_datetime_object(self, dialog):
        dt = datetime(2025, 6, 15, 10, 30)
        result = dialog._format_datetime(dt)
        assert "15" in result
        assert "2025" in result

    def test_none_value(self, dialog):
        result = dialog._format_datetime(None)
        assert "nije" in result.lower() or "postavljeno" in result.lower()

    def test_empty_string(self, dialog):
        result = dialog._format_datetime("")
        assert isinstance(result, str)

    def test_invalid_format(self, dialog):
        result = dialog._format_datetime("not-a-date")
        assert isinstance(result, str)


class TestParseModification:
    def test_note_modification(self, dialog):
        result = dialog.parse_modification("Bilješka: službeni put")
        assert result is not None
        param, value = result
        assert param == "Note"
        assert "službeni put" in str(value)

    def test_from_time(self, dialog):
        result = dialog.parse_modification("od 10h")
        assert result is not None
        param, value = result
        assert param == "FromTime"
        assert "10" in str(value)

    def test_to_time(self, dialog):
        result = dialog.parse_modification("Do: 17:30")
        assert result is not None
        param, value = result
        assert param == "ToTime"
        assert "17" in str(value)

    def test_mileage(self, dialog):
        result = dialog.parse_modification("Km: 15000")
        assert result is not None
        param, value = result
        assert param == "Value"

    def test_subject(self, dialog):
        result = dialog.parse_modification("Naslov: prijava kvara")
        assert result is not None
        param, value = result
        assert param == "Subject"

    def test_no_match_returns_none(self, dialog):
        result = dialog.parse_modification("hello world")
        assert result is None

    def test_case_insensitive(self, dialog):
        result = dialog.parse_modification("BILJEŠKA: text")
        assert result is not None


class TestProcessModificationValue:
    def test_time_with_colon(self, dialog):
        result = dialog._process_modification_value("FromTime", "10:30")
        assert "10:30" in str(result)

    def test_time_with_dot(self, dialog):
        result = dialog._process_modification_value("FromTime", "10.30")
        assert "10" in str(result) and "30" in str(result)

    def test_time_hour_only(self, dialog):
        result = dialog._process_modification_value("ToTime", "9")
        assert "09" in str(result) or "9" in str(result)

    def test_mileage_integer(self, dialog):
        result = dialog._process_modification_value("Value", "15000")
        assert result == 15000

    def test_mileage_with_separator(self, dialog):
        result = dialog._process_modification_value("Value", "15.000")
        assert result == 15000

    def test_text_field_passthrough(self, dialog):
        result = dialog._process_modification_value("Note", "some note")
        assert result == "some note"


class TestGetOperationType:
    def test_booking_tool(self, dialog):
        result = dialog._get_operation_type("post_BookingCalendar")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mileage_tool(self, dialog):
        result = dialog._get_operation_type("post_AddMileage")
        assert isinstance(result, str)

    def test_delete_tool(self, dialog):
        result = dialog._get_operation_type("delete_Something")
        assert isinstance(result, str)

    def test_unknown_tool(self, dialog):
        result = dialog._get_operation_type("xyz_unknown")
        assert isinstance(result, str)


class TestGenerateConfirmationMessage:
    def test_generates_message(self, dialog):
        params = [
            ParameterDisplay(
                name="VehicleId",
                display_name="Vozilo",
                value="VW Passat",
                display_value="VW Passat (ZG-1234)",
                is_required=True,
                is_editable=False,
                description="Vehicle"
            ),
            ParameterDisplay(
                name="FromTime",
                display_name="Od",
                value="2025-06-15T10:00:00",
                display_value="15.06.2025. 10:00",
                is_required=True,
                is_editable=True,
                description="Start time"
            ),
        ]
        result = dialog.generate_confirmation_message("post_BookingCalendar", params)
        assert isinstance(result, str)
        assert "Da" in result or "Ne" in result
        assert len(result) > 50

    def test_empty_params(self, dialog):
        result = dialog.generate_confirmation_message("post_Test", [])
        assert isinstance(result, str)


class TestGenerateUpdateMessage:
    def test_generates_update(self, dialog):
        result = dialog.generate_update_message("Note", "old note", "new note")
        assert "new note" in result
        assert isinstance(result, str)


class TestFormatValue:
    def test_none_value(self, dialog):
        result = dialog._format_value("Any", None, {})
        assert "prazno" in result.lower() or result == str(None)

    def test_vehicle_id_with_context(self, dialog):
        context = {
            "vehicles": [
                {"VehicleId": "v-123", "FullVehicleName": "VW Passat", "LicencePlate": "ZG-1234"}
            ]
        }
        result = dialog._format_value("VehicleId", "v-123", context)
        assert "VW Passat" in result or "v-123" in result

    def test_long_string_truncated(self, dialog):
        long_value = "x" * 100
        result = dialog._format_value("Description", long_value, {})
        assert len(result) <= 60  # 47 + "..." + some buffer
