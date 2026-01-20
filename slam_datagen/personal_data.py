from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TypeAlias

from faker import Faker
from faker.providers import (automotive, bank, company, credit_card, internet, misc,
                             passport, person, phone_number, profile)

from slam_datagen.utils.typing import NestedStrDict

ProfileValue: TypeAlias = str | tuple[Decimal, Decimal] | list[str] | date


@dataclass
class PersonalData:
    unique_identifiers: dict[str, str]
    attributes: dict[str, NestedStrDict]


class PersonalDataGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self.fake = Faker(["en_US"])
        if seed is not None:
            self.fake.seed_instance(seed)
        self.fake.add_provider(automotive)
        self.fake.add_provider(bank)
        self.fake.add_provider(company)
        self.fake.add_provider(credit_card)
        self.fake.add_provider(internet)
        self.fake.add_provider(misc)
        self.fake.add_provider(passport)
        self.fake.add_provider(person)
        self.fake.add_provider(phone_number)
        self.fake.add_provider(profile)

    def generate(self, n: int) -> list[PersonalData]:
        data: list[PersonalData] = []
        for _ in range(n):
            email = self.fake.ascii_email()
            unique_identifiers = {
                "name": " ".join([self.fake.first_name(), self.fake.last_name()]),
                "ssn": self._generate_from_profile("ssn"),
            }

            attributes: dict[str, NestedStrDict] = {
                "profile": {
                    "sex": self._generate_from_profile("sex"),
                    "blood_group": self._generate_from_profile("blood_group"),
                    "date_of_birth": self.fake.passport_dob().isoformat(),
                    "photo": self.fake.image_url(),
                },
                "car": {
                    "license_plate": self.fake.license_plate(),
                    "vin": self.fake.vin(),
                },
                "bank_account": {
                    "bank_country": self.fake.bank_country(),
                    "bban": self.fake.bban(),
                    "aba": self.fake.aba(),
                    "iban": self.fake.iban(),
                    "swift": self.fake.swift(),
                    "credit_card": {
                        "expire": self.fake.credit_card_expire(),
                        "number": self.fake.credit_card_number(),
                        "provider": self.fake.credit_card_provider(),
                        "security_code": self.fake.credit_card_security_code(),
                    },
                },
                "contacts": {
                    "phone": self.fake.phone_number(),
                    "email": email,
                    "website": self.fake.url(),
                    "telegram": self.fake.user_name(),
                    "social_networks": {
                        "vk": self.fake.user_name(),
                        "twitter": self.fake.user_name(),
                        "linkedin": self.fake.user_name(),
                        "facebook": self.fake.user_name(),
                    },
                },
                "internet_access_point": {
                    "ipv4": self.fake.ipv4(),
                    "ipv6": self.fake.ipv6(),
                    "mac": self.fake.mac_address(),
                },
                "passports": {
                    "national_passport_number": self.fake.passport_number(),
                    "international_passport_number": self.fake.passport_number(),
                },
                "work": {
                    "location": self._generate_from_profile("current_location"),
                    "company": self._generate_from_profile("company"),
                    "address": self.fake.address(),
                },
                "home": {
                    "address": self.fake.address(),
                    "location": self._generate_from_profile("current_location"),
                },
            }

            data.append(
                PersonalData(
                    unique_identifiers=unique_identifiers, attributes=attributes
                )
            )

        return data

    def _generate_from_profile(self, field: str) -> str:
        profile_value = self.fake.profile(fields=[field])[field]
        if isinstance(profile_value, tuple):
            latitude, longitude = profile_value
            return f"({float(latitude):.6f}, {float(longitude):.6f})"
        if isinstance(profile_value, list):
            return ", ".join(map(str, profile_value))
        if isinstance(profile_value, date):
            return profile_value.isoformat()
        return str(profile_value)

    # print(fake.url())  # internet
    # print(fake.image_url())  # internet
    # print(fake.password())  # misc
    # print(fake.uuid4())  # misc
    # print(fake.passport_dob())  # passport
