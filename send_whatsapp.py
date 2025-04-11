import argparse
import os.path

from requests import post


def is_phone(s):
    if len(str(s)) < 6:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False
    except:
        return False


def is_valid_israel_phone(s):
    if not is_phone(s):
        return False
    if len(s) == 9 and (str(s).startswith("5") or str(s).startswith("7")):
        return True
    if len(s) == 10 and (str(s).startswith("05") or str(s).startswith("07")):
        return True
    if len(s) == 8 and (
            str(s).startswith("2") or str(s).startswith("3") or str(s).startswith("4") or str(s).startswith(
        "8") or str(s).startswith("9")):
        return True
    if len(s) == 9 and (
            str(s).startswith("02") or str(s).startswith("03") or str(s).startswith("04") or str(s).startswith(
        "08") or str(s).startswith("09")):
        return True
    return False


def to_integer(phone_number, prefix="972"):
    if phone_number == "anonymous":
        return phone_number
    str_phone_number = phone_number
    if isinstance(phone_number, int):
        str_phone_number = str(phone_number)

    if len(str_phone_number) <= 6:
        return str_phone_number

    clean_phone_number = to_digits(str_phone_number)

    if clean_phone_number[0:1] == "0" and clean_phone_number[0:2] != "00":
        return "%s" % clean_phone_number[1:]
    if clean_phone_number[0:len(prefix)] == prefix:
        return "%s" % clean_phone_number[len(prefix):]
    return clean_phone_number


def to_digits(phone_number):
    """
    eliminates unwanted characters from phone number
    :param phone_number: unicode
    :return: unicode
    """
    return ''.join(x for x in phone_number.__str__() if x.isdigit())


def to_e164(phone_number, prefix="972", local_prefix=""):
    if phone_number == "anonymous":
        return phone_number
    str_phone_number = phone_number
    if isinstance(phone_number, int):
        str_phone_number = str(phone_number)

    clean_phone_number = to_digits(str_phone_number.__str__())

    if prefix == "972" and local_prefix != "" and len(to_integer(phone_number)) == 7:
        return "%s%s%s" % (prefix, local_prefix.replace("0", ""), to_integer(phone_number))

    if len(clean_phone_number) < 5:
        return clean_phone_number
    if str_phone_number.__str__()[0:1] == "+":
        return clean_phone_number
    if clean_phone_number[0:2] == "00":
        return str(int(clean_phone_number))
    if clean_phone_number.__str__()[0:1] == "0":
        return "%s%s" % (prefix, clean_phone_number[1:])
    if prefix == "972":
        if clean_phone_number[0:3] != prefix and is_valid_israel_phone(clean_phone_number):
            return "%s%s" % (prefix, clean_phone_number)
    else:
        if clean_phone_number[0:3] != prefix:
            return "%s%s" % (prefix, clean_phone_number)

    return clean_phone_number


ap = argparse.ArgumentParser()
ap.add_argument("-a", "--attachment", required=False, help="file attachment")
ap.add_argument("-p", "--phone", required=True, help="phone number")
ap.add_argument("-m", "--message", required=True, help="mesage")
ap.add_argument("-s", "--sender", required=False, help="sender")
ap.add_argument("-u", "--url", required=True, help="url", default="master.meteor-comm.com")

args = vars(ap.parse_args())

files = {}
if args["attachment"] and os.path.exists(args["attachment"]):
    files["attachment1"] = open(args["attachment"], "rb")

params = {"message": args["message"],
          "recipient_phone": to_e164(args["phone"]),
          "ignore_replies": True,
          }

if args["sender"]:
    params["sender"] = args["sender"]

result = post("http://%s/api/whastapp_messages/" % args["url"], data=params, files=files)
print(result.text)
